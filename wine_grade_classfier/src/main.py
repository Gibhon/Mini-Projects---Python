from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

eps: float = 1e-7


def load_data(file_path: str) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter=";", skiprows=1)
    features_np = data[:, :-1]
    labels_np = data[:, -1]

    conditions = [(labels_np <= 5), (labels_np == 6), (labels_np >= 7)]
    choices = [0, 1, 2]
    labels_np_collapsed = np.select(conditions, choices)

    features_np_final = features_np.astype(np.float32)
    labels_np_final = labels_np_collapsed.astype(np.int64)

    return features_np_final, labels_np_final


class WineDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        features = torch.tensor(
            features, dtype=torch.float32
        )  # dtype not necessary here tho
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.features = (features - features.mean(dim=0)) / features.std(dim=0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class WineClassifier(nn.Module):
    def __init__(self, n_inputs, n_neurons1, n_neurons2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=n_inputs, out_features=n_neurons1)
        self.layer2 = nn.Linear(in_features=n_neurons1, out_features=n_neurons2)
        self.layer3 = nn.Linear(n_neurons2, 3)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, inputs):
        inputs = self.layer1(inputs)
        inputs = self.relu(inputs)
        inputs = self.dropout1(inputs)
        inputs = self.layer2(inputs)
        inputs = self.layer3(self.dropout2(self.relu(inputs)))
        return inputs


def train_model(n_epoch, model, train_set_loader, loss_fn, optimizer, path):
    min_loss = 100
    for epoch in range(n_epoch):
        total_loss = 0
        n_accurate = 0
        n_total = 0
        for x_batch, y_batch in train_set_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction_dim3 = model.forward(x_batch)
            prediction_final = torch.argmax(prediction_dim3, dim=1)
            loss = loss_fn(prediction_dim3, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_accurate += (prediction_final == y_batch).sum().item()
            n_total += prediction_final.size(0)
        current_loss = total_loss / len(train_set_loader)
        if min_loss > current_loss:
            min_loss = current_loss
            min_epoch = epoch
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

        if epoch % (n_epoch / 10) == 0:
            print(f"Loss : {current_loss}")
            print(f"Accuracy: {n_accurate / n_total}")
    print(f"MinLoss: {min_loss}")
    print(min_epoch)
    torch.save(checkpoint, path)
    return min_loss
    # torch.save(
    #     model.state_dict(),
    #     r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\wine_grade_classfier\data\weights.pth",
    # )


def val_model(model, val_set_loader, loss_fn, path):
    # 1. Load the full checkpoint dictionary
    checkpoint = torch.load(path, weights_only=True)

    # 2. Extract and load ONLY the model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    total_loss = 0
    n_accurate = 0
    n_total = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_set_loader:
            prediction_dim3 = model.forward(x_batch)
            prediction_final = torch.argmax(prediction_dim3, dim=1)
            loss = loss_fn(prediction_dim3, y_batch)
            total_loss += loss.item()
            n_accurate += (prediction_final == y_batch).sum().item()
            n_total += prediction_final.size(0)
    print("--------------------------------------------")
    print(f"loss : {total_loss / len(val_set_loader)}")
    print(f"Accuracy: {n_accurate / n_total}")


features_np, labels_np = load_data(
    Path(__file__).resolve().parent.parent / "data" / "winequality-red.csv"
)
data_set = WineDataset(features_np, labels_np)
train_set, val_set = random_split(
    data_set, lengths=[0.8, 0.2], generator=torch.Generator().manual_seed(42)
)

train_set_loader = DataLoader(train_set, shuffle=True, batch_size=32)
val_set_loader = DataLoader(val_set, shuffle=False, batch_size=32)

torch.manual_seed(42)
model = WineClassifier(11, 16, 16)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

model_location = Path(__file__).resolve().parent.parent / "data" / "model.pth"

min_loss = train_model(90, model, train_set_loader, loss_fn, optimizer, model_location)

val_model(model, val_set_loader, loss_fn, model_location)

# loss : 0.8093988776206971
# Accuracy: 0.6489028213166145
