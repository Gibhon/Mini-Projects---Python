import torch
from torch import nn

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
    
model = StudentClassifier(2, 4)

print(model)