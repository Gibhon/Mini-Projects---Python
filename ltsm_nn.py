import torch
import torch.nn as nn


class LTSMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=1, batch_first=True):
        super().__init__()
        self.ltsm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

    def forward(self, inputs):
        out, (hn, cn) = self.ltsm(inputs)

        return out, hn, cn


batch = 4
seq_len = 6
input_size = 10

dummy_input = torch.randn(batch, seq_len, input_size)

model = LTSMModel()
output, hn, cn = model(dummy_input)

print(output.shape)
print(hn.shape)
print(cn.shape)
