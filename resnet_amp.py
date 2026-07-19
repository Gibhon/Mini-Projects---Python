import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as cp
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_cp=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.use_cp = use_cp

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs):
        identity = self.shortcut(inputs)

        def residual_segment(inputs):
            outputs = self.relu(self.bn1(self.conv1(inputs)))
            outputs = self.bn2(self.conv2(outputs))
            return outputs

        if self.use_cp:
            outputs = cp(residual_segment, inputs, use_reentrant=False)
        else:
            outputs = residual_segment(inputs)

        outputs += identity

        return self.relu(outputs)


class CNNModel(nn.Module):
    def __init__(self, n_inputs=4096, n_neurons=256, use_cp=True):
        super().__init__()

        self.layer1 = nn.Linear(n_inputs, n_neurons)
        self.layer2 = nn.Linear(n_neurons, 10)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)

        self.res1 = ResidualBlock(32, 32, use_cp)
        self.res2 = ResidualBlock(64, 64, use_cp)

    def forward(self, inputs):
        outputs = self.maxpool(self.res1(self.relu(self.bn1(self.conv1(inputs)))))
        outputs = self.maxpool(self.res2(self.relu(self.bn2(self.conv2(outputs)))))
        outputs = self.dropout(self.flatten(outputs))
        outputs = self.layer2(self.relu(self.layer1(outputs)))

        return outputs


def load_data():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    )
    full_train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    train_set, val_set = random_split(
        full_train_set,
        lengths=[40000, 10000],
        generator=torch.Generator().manual_seed(42),
    )
    train_set_loader = DataLoader(train_set, shuffle=True, batch_size=64, num_workers=4)
    val_set_loader = DataLoader(val_set, shuffle=False, batch_size=64, num_workers=3)
    test_set_loader = DataLoader(test_set, shuffle=False, batch_size=64, num_workers=3)

    return train_set_loader, val_set_loader, test_set_loader


def train_model(
    model, n_epoch, train_set_loader, loss_fn, optimizer, device, scheduler, with_amp
):
    start = time.time()
    total_loss = 0
    n_accurate = 0
    n_total = 0
    scaler = torch.amp.GradScaler(device.type) if with_amp else None
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(n_epoch):
        for x_batch, y_batch in train_set_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                predictions = model(x_batch)
                loss = loss_fn(predictions, y_batch)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                loss.backward()
                optimizer.step()
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

            total_loss += loss.item()
            n_total += predictions.size(0)
            _, pred_labels = torch.max(predictions, 1)
            n_accurate += (pred_labels == y_batch).sum().item()
        scheduler.step()

        print(f"Max_memory: {peak_mem:.2f}MB")
    final_time = time.time() - start
    Accuracy = (n_accurate / n_total) * 100
    return (final_time, Accuracy)


def val_model(model, val_set_loader, loss_fn, device):
    total_loss = 0
    n_total = 0
    n_accurate = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_set_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)

            total_loss += loss.item()
            n_total += prediction.size(0)
            _, prediction_label = torch.max(prediction, 1)
            n_accurate += (prediction_label == y_batch).sum().item()

    print(f"Loss : {total_loss / len(val_set_loader)}")
    print(f"Accuracy: {n_accurate / n_total}")


def test_model(model, test_set_loader, device):
    n_total = 0
    n_accurate = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_set_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch)

            n_total += prediction.size(0)
            _, prediction_label = torch.max(prediction, 1)
            n_accurate += (prediction_label == y_batch).sum().item()

    print(f"Accuracy: {n_accurate / n_total}")


if __name__ == "__main__":
    train_set_loader, val_set_loader, test_set_loader = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model1 = CNNModel().to(device)
    torch.manual_seed(42)
    model2 = CNNModel(use_cp=False).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), weight_decay=1e-4, lr=1e-3)
    optimizer2 = torch.optim.Adam(model2.parameters(), weight_decay=1e-4, lr=1e-3)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.1)

    print("Model with checkpoint")
    model_withcp = train_model(
        model1, 5, train_set_loader, loss_fn, optimizer1, device, scheduler1, False
    )
    print("Model without checkpoint")
    model_withoutcp = train_model(
        model2, 5, train_set_loader, loss_fn, optimizer2, device, scheduler2, False
    )
    print("With_CP")
    print(f"Time:{model_withcp[0]} \n Accuracy: {model_withcp[1]}")
    print("Without_CP")
    print(f"Time:{model_withoutcp[0]} \n Accuracy: {model_withoutcp[1]}")
