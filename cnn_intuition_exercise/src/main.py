import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets

class CNNModel(nn.Module):
    def __init__(self, n_inputs : int = 1568, n_neurons : int = 128):
        super().__init__()
        self.layer1 = nn.Linear(in_features = n_inputs, out_features = n_neurons)
        self.layer2 = nn.Linear(in_features = n_neurons, out_features = 10)
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, inputs : torch.Tensor) -> torch.Tensor:
        inputs = self.max_pool(self.relu(self.conv1(inputs)))
        inputs = self.max_pool(self.relu(self.conv2(inputs)))
        inputs = self.flatten(inputs)
        inputs = self.relu(self.layer1(inputs))
        return self.layer2(inputs)
    
def load_data():
    transform = transforms.ToTensor()
    full_train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
    )
    train_set, val_set = random_split(full_train_dataset, lengths=[50000, 10000], generator=torch.Generator().manual_seed(42))
    train_set_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    val_set_loader = DataLoader(val_set, shuffle=False, batch_size=32)
    return train_set_loader, val_set_loader

def train_model(n_epoch, model, train_set_loader, optimizer, loss_fn, device):
    for epoch in range(n_epoch):
        total_loss = 0
        n_total = 0
        n_accurate = 0
        for x_batch, y_batch in train_set_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none = True)
            prediction = model.forward(x_batch)
            loss = loss_fn(prediction , y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_total += prediction.size(0)
            _, predicted_labels = torch.max(prediction, 1)
            n_accurate += (predicted_labels == y_batch).sum().item()
        if (epoch % 1 == 0):
            print(f"Loss : {total_loss/len(train_set_loader)}")
            print(f"Accuracy: {n_accurate / n_total}")
    torch.save(model.state_dict(), r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\cnn_intuition_exercise\data\model.pth")

def val_model(model, val_set_loader, loss_fn, device):
    model.load_state_dict(torch.load(r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\cnn_intuition_exercise\data\model.pth"))
    total_loss = 0
    n_total = 0
    n_accurate = 0
    with torch.no_grad():
        for x_batch, y_batch in val_set_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model.forward(x_batch)
            loss = loss_fn(prediction , y_batch)
            total_loss += loss.item()
            n_total += prediction.size(0)
            _, predicted_labels = torch.max(prediction, 1)
            n_accurate += (predicted_labels == y_batch).sum().item()
    print(f"Loss : {total_loss/len(val_set_loader)}")

    print(f"Accuracy: {n_accurate / n_total}")
    
if __name__ == "__main__":
    train_set_loader, val_set_loader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = CNNModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)
    train_model(5, model, train_set_loader, optimizer, loss_fn, device)
    val_model(model, val_set_loader, loss_fn, device)