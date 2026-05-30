import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

eps = 1e-7

data_np = np.loadtxt(r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\diabetes_test\data\pima.csv", delimiter=",")
features_np = data_np[:, :-1]
labels_np = data_np[:, -1:]

features_tensor = torch.tensor(features_np, dtype = torch.float32)
features_scaled = (features_tensor - features_tensor.mean(dim = 0)) / (features_tensor.std(dim = 0) + eps)
labels = torch.tensor(labels_np, dtype= torch.float32)

class DiabetesDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
data_set = DiabetesDataset(features_scaled, labels)
train_set, val_set = random_split(data_set, lengths = [0.8, 0.2], generator = torch.Generator().manual_seed(42))

train_set_loader = DataLoader(train_set, shuffle=True, batch_size=32)
val_set_loader = DataLoader(val_set, shuffle=False, batch_size=32)

class DiabetesClassifier(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()
        self.layer1 = nn.Linear(in_features=n_inputs, out_features=n_neurons)
        self.layer2 = nn.Linear(in_features=n_neurons, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        input = self.layer1(input)
        input = self.relu(input)
        input = self.layer2(input)
        return self.sigmoid(input)
    
torch.manual_seed(42)
model = DiabetesClassifier(8, 10) 
#I used the scaling method to get to this number of neurons but it is learning way to bad to use only 6 let me try 10 too slow still let me try 12 then if not ill try another optimizer yah still in range of 7-8 ill switch up. 

loss_fn = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
# Replace your SGD line with this
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(400):
    epoch_loss = 0
    val_loss = 0
    accurate_guess_counter = 0
    total_counter = 0
    
    for x_batch, y_batch in train_set_loader:
        optimizer.zero_grad(set_to_none=True)
        prediction_training = model.forward(x_batch)
        loss = loss_fn(prediction_training, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    with torch.no_grad():
        for x_batch, y_batch in val_set_loader:
            prediction_val = model.forward(x_batch)
            loss = loss_fn(prediction_val, y_batch)
            val_loss += loss.item()
            # if(prediction_val == y_batch):
            #     accurate_guess_counter += 1
            # matches = (prediction_val == y_batch)
            prediction_labels = (prediction_val >= 0.5).float()
            prediction_labels = prediction_labels.view(-1)
            y_batch = y_batch.view(-1)
            matches = (prediction_labels == y_batch)
            accurate_guess_counter += matches.sum().item()
            total_counter += y_batch.size(0)
                
    if(epoch % 100 == 0):
        print(f"Epoch : {epoch} \n Epoch_loss : {epoch_loss / len(train_set_loader)} \n Val_Loss : {val_loss/ len(val_set_loader)} \n Accuracy : {accurate_guess_counter/total_counter}")
        
        
# After training, save the model weights to a file using torch.save
# Write a separate loading section that loads the weights back into a fresh model
# Run inference on the val set with the loaded model and confirm accuracy matches
torch.save(model.state_dict(), r"C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\diabetes_test\data\diabetes_weights.pth")

model.load_state_dict(torch.load(r'C:\Users\Acer\Documents\Code-Main\Python\Mini-Projects\diabetes_test\data\diabetes_weights.pth'))
model.eval()



with torch.no_grad():
    accurate_guess_counter = 0
    total_counter = 0
    for x_batch, y_batch in val_set_loader:
        prediction_val = model.forward(x_batch)
        loss = loss_fn(prediction_val, y_batch)
        val_loss += loss.item()
            # if(prediction_val == y_batch):
            #     accurate_guess_counter += 1
            # matches = (prediction_val == y_batch)
        prediction_labels = (prediction_val >= 0.5).float()
        prediction_labels = prediction_labels.view(-1)
        y_batch = y_batch.view(-1)
        matches = (prediction_labels == y_batch)
        accurate_guess_counter += matches.sum().item()
        total_counter += y_batch.size(0)
    print(f"Epoch : {epoch} \n Epoch_loss : {epoch_loss / len(train_set_loader)} \n Val_Loss : {val_loss/ len(val_set_loader)} \n Accuracy : {accurate_guess_counter/total_counter}")
        
