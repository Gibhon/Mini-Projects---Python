import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import time



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )    
        
    
    def forward(self, inputs):
        identity = self.shortcut(inputs)
        
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        outputs += identity
        
        return self.relu(outputs)
        
        

class ResnetCNN(nn.Module):
    def __init__(self, n_inputs = 4096, n_neurons = 256):
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
        self.dropout = nn.Dropout(p=0.35)
        
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(64, 64)
        
    
    def forward(self, inputs):
        outputs = self.maxpool(self.res1(self.relu(self.bn1(self.conv1(inputs)))))
        outputs = self.maxpool(self.res2(self.relu(self.bn2(self.conv2(outputs)))))
        outputs = self.dropout(self.flatten(outputs))
        outputs = self.layer2(self.relu(self.layer1(outputs)))
        
        return outputs



def load_data():
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
    ])
    full_train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_set, val_set = random_split(full_train_set, lengths = [40000, 10000], generator=torch.Generator().manual_seed(42))
    train_set_loader = DataLoader(train_set, shuffle=True, batch_size=64,  num_workers=4)
    val_set_loader = DataLoader(val_set, shuffle=False, batch_size=64)
    test_set_loader = DataLoader(test_set, shuffle=False, batch_size=64)
    
    return train_set_loader, val_set_loader, test_set_loader


def train_model(model, n_epoch, train_set_loader, loss_fn, optimizer, device, scheduler):
    for epoch in range(n_epoch):
        total_loss = 0
        n_accurate = 0
        n_total = 0
        for x_batch, y_batch in train_set_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_total += predictions.size(0)
            _, prediction_labels = torch.max(predictions, 1)
            n_accurate += (prediction_labels == y_batch).sum().item()
        scheduler.step()
        print(f"epoch: {epoch}")
        print(f"Loss : {total_loss/len(train_set_loader)}")
        print(f"Accuracy: {n_accurate / n_total}")
    
    torch.save(model.state_dict(), r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\Resnet_CNN\data\model.pth")
    
    
def val_model(model, val_set_loader, loss_fn, device):
    model.load_state_dict(torch.load(r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\Resnet_CNN\data\model.pth", map_location=device))
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
            
    print(f"Loss : {total_loss/len(val_set_loader)}")
    print(f"Accuracy: {n_accurate / n_total}")
    

def test_model(model, test_set_loader, device):
        model.load_state_dict(torch.load(r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\Resnet_CNN\data\model.pth", map_location=device))
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
    start = time.time()
    
    train_set_loader, val_set_loader, test_set_loader = load_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = ResnetCNN().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    train_model(model, 60, train_set_loader, loss_fn, optimizer, device, scheduler)
    val_model(model, val_set_loader, loss_fn, device)
    test_model(model, test_set_loader, device)
    
    print(f"Time: {time.time() - start:.1f}s")


