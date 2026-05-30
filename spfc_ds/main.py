import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

eps = 1e-7

x = torch.tensor(
    [
        [10.5, 95.0],
        [8.0, 90.0],
        [12.0, 85.0],
        [2.0, 50.0],
        [4.5, 60.0],
        [1.0, 40.0],
    ],
    dtype=torch.float32,
)
mean_x = x.mean(dim=0)
std_x = x.std(dim=0)
x_scaled = (x - mean_x) / (std_x + eps)

y = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)  #The defualt type itself is float32 so it is kindof redundant


class StudentDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        
        return x, y

dataset = StudentDataset(x_scaled, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
class StudentClassifier(nn.Module):
    def __init__(self, n_inputs : int, n_neurons : int):
        super().__init__() 
        self.layer1 = nn.Linear(in_features= n_inputs, out_features= n_neurons) 
        self.layer2 = nn.Linear(n_neurons, 1) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, inputs : torch.Tensor):
        inputs = self.layer1(inputs)
        inputs = self.relu(inputs)
        inputs = self.layer2(inputs)
        return self.sigmoid(inputs)
    
torch.manual_seed(42)
model = StudentClassifier(2, 4)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1)

for epoch in range(100):
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad(set_to_none= True)
    
        prediction = model(x_batch)
    
        loss = loss_fn(prediction, y_batch.unsqueeze(1))
    
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if(epoch % 10 == 0):
        print(epoch_loss)
        
    
    
        
with torch.no_grad():
    predictions = model(x_scaled)
    print(f"Predictions : {torch.round(predictions).squeeze(1)}")
    print(f"Actual: {y}")
    