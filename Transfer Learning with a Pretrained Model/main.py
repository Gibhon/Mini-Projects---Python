import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time

def load_model():
    return resnet18(weights = ResNet18_Weights.DEFAULT)  #copied from ai and i prolly cant load a model like this for other models or even this model on my own


def freeze_layers(model):
    layers = list(model.children())
    last_layer = layers[-1]
    
    for param in model.parameters():
        param.requires_grad = False
    for param in last_layer.parameters():
        param.requires_grad = True  #copied but i can do it myself now this whole section
        
        
def replace_layer(model):
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    

def load_data():
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize(224),
    transforms.ToTensor()
    ])
    full_train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_set, val_set = random_split(full_train_set, lengths = [40000, 10000], generator=torch.Generator().manual_seed(42))
    train_set_loader = DataLoader(train_set, shuffle=True, batch_size=64,  num_workers=4)
    val_set_loader = DataLoader(val_set, shuffle=False, batch_size=64)
    test_set_loader = DataLoader(test_set, shuffle=False, batch_size=64)
    
    return train_set_loader, val_set_loader, test_set_loader
        

def train_model(model, n_epoch, train_set_loader, optimizer, loss_fn, device, scheduler):
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
    
    torch.save(model.state_dict(), r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\Transfer Learning with a Pretrained Model\data\model.pth")
    
    
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
            
    print(f"Loss : {total_loss/len(val_set_loader)}")
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
    start = time.time()
    
    train_set_loader, val_set_loader, test_set_loader = load_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = load_model()
    freeze_layers(model)
    replace_layer(model)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), weight_decay=1e-4, lr=1e-3)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001) this is what i copied and used after editing
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
        
    train_model(model, 10, train_set_loader, optimizer, loss_fn, device, scheduler)
    val_model(model, val_set_loader, loss_fn, device)
    test_model(model, test_set_loader, device)
    
    print(f"Time: {time.time() - start:.1f}s")
