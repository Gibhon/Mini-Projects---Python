import torch
from torch import nn

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

for epoch in range(10001):
    optimizer.zero_grad(set_to_none= True) # it help save memory of internal grads and set to none just makes the default to 0 setting to just delete the internal grads
    
    # prediction = model.forward(x)  Looks like it is related to wrapper and something called __call__ which makes the traditional method calling inaccurate
    prediction = model(x_scaled)
    
    loss = loss_fn(prediction, y.unsqueeze(1))
    
    loss.backward()
    optimizer.step()
    
    if(epoch % 1000 == 0):
        print(f"Loss at {epoch} epoch : {loss}")
    