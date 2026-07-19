import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets

class CiferModel(nn.Module):
    def __init__(self, n_inputs = 4096, n_neurons = 256):
        super().__init__()
        
        self.layer1 = nn.Linear(n_inputs, n_neurons)
        self.layer2 = nn.Linear(n_neurons, 10)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, inputs):
        inputs = self.maxpool(self.relu(self.conv1(inputs)))
        inputs = self.maxpool(self.relu(self.conv2(inputs)))
        inputs = self.flatten(inputs)
        inputs = self.dropout(inputs)
        inputs = self.layer2(self.relu(self.layer1(inputs)))
        
        return inputs
    
    
    
def load_data():
    transform = transforms.ToTensor()
    full_train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_set, val_set = random_split(full_train_set, lengths = [40000, 10000], generator=torch.Generator().manual_seed(42))
    train_set_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    val_set_loader = DataLoader(val_set, shuffle=False, batch_size=32)
    test_set_loader = DataLoader(test_set, shuffle=False, batch_size=32)
    
    return train_set_loader, val_set_loader, test_set_loader


def train_model(model, n_epoch, train_set_loader, loss_fn, optimizer, device):
    for epoch in range(n_epoch):
        total_loss = 0
        n_accurate = 0
        n_total = 0
        for x_batch, y_batch in train_set_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none = True)
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_total += prediction.size(0)
            _, prediction_label = torch.max(prediction, 1)
            n_accurate += (prediction_label == y_batch).sum().item()
            
        print(f"Loss : {total_loss/len(train_set_loader)}")
        print(f"Accuracy: {n_accurate / n_total}")
        torch.save(model.state_dict(), r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\CIFAR_10_CNN\data\model.pth")


def val_model(model, val_set_loader, loss_fn, device):
    model.load_state_dict(torch.load(r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\CIFAR_10_CNN\data\model.pth", map_location=device))
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
        model.load_state_dict(torch.load(r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\CIFAR_10_CNN\data\model.pth", map_location=device))
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
    model = CiferModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    train_model(model, 5, train_set_loader, loss_fn, optimizer, device)
    val_model(model, val_set_loader, loss_fn, device)
    test_model(model, test_set_loader, device)