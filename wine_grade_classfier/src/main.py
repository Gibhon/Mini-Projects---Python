import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

eps : float = 1e-7

def load_data(file_path : str) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter=";", skiprows=1)
    features_np = data[: , :-1]
    labels_np = data[:, -1]
    
    conditions = [
    (labels_np <= 5),
    (labels_np == 6),
    (labels_np >= 7)
    ]
    choices = [0, 1, 2]
    labels_np_collapsed = np.select(conditions, choices)
    
    features_np_final = features_np.astype(np.float32)
    labels_np_final = labels_np_collapsed.astype(np.int64)
    
    return features_np_final, labels_np_final


class WineDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        features = torch.tensor(features, dtype=torch.float32) #dtype not necessary here tho
        self.labels = torch.tensor(labels, dtype = torch.long)
        self.features = (features - features.mean(dim = 0)) / features.std(dim = 0)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
class WineClassifier(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()
        self.layer1 = nn.Linear(in_features=  n_inputs, out_features= n_neurons)
        self.layer2 = nn.Linear(in_features= n_neurons, out_features= 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)   
    def forward(self, inputs):
        inputs = self.layer1(inputs)
        inputs = self.relu(inputs)
        inputs = self.dropout(inputs)
        inputs = self.layer2(inputs)
        return inputs
    
def train_model(n_epoch, model, train_set_loader, loss_fn, optimizer):
    for epoch in range(n_epoch):
        total_loss = 0
        n_accurate = 0
        n_total = 0
        for x_batch, y_batch in train_set_loader:
            optimizer.zero_grad(set_to_none = True)
            prediction_dim3 = model.forward(x_batch)
            prediction_final = torch.argmax(prediction_dim3, dim = 1)
            loss = loss_fn(prediction_dim3, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_accurate += (prediction_final == y_batch).sum().item()
            n_total += prediction_final.size(0)
        if (epoch % (n_epoch/10) == 0):
            print(f"Loss : {total_loss/len(train_set_loader)}")
            print(f"Accuracy: {n_accurate / n_total}")
    torch.save(model.state_dict(), r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\wine_grade_classfier\data\weights.pth")

def val_model(model, val_set_loader, loss_fn):
    model.load_state_dict(torch.load(r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\wine_grade_classfier\data\weights.pth"))
    total_loss = 0
    n_accurate = 0
    n_total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_set_loader:
            prediction_dim3 = model.forward(x_batch)
            prediction_final = torch.argmax(prediction_dim3, dim = 1)
            loss = loss_fn(prediction_dim3, y_batch)
            total_loss += loss.item()
            n_accurate += (prediction_final == y_batch).sum().item()
            n_total += prediction_final.size(0)
            
    print(f"loss : {total_loss/ len(val_set_loader)}")
    print(f"Accuracy: {n_accurate / n_total}")


features_np, labels_np = load_data(r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\wine_grade_classfier\data\winequality-red.csv")
data_set = WineDataset(features_np, labels_np)
train_set, val_set = random_split(data_set, lengths=[.8, .2], generator=torch.Generator().manual_seed(42))

train_set_loader = DataLoader(train_set, shuffle=True, batch_size=32)
val_set_loader = DataLoader(val_set, shuffle=False, batch_size=32)

torch.manual_seed(42)
model = WineClassifier(11, 20)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay=1e-4)

# train_model(100, model, train_set_loader, loss_fn, optimizer)

val_model(model, val_set_loader, loss_fn)

