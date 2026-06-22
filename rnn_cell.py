import torch
import torch.nn as nn


class RNNFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            h = torch.tanh(self.W_xh(x_t) + self.W_hh(h))
        return self.out(h)


def fake_data_creator():
    starts = torch.randint(0, 50, (100,))
    steps = torch.randint(1, 10, (100,))
    data_sequence = []
    data_target = []
    for i in range(len(starts)):
        sequence = []
        target = starts[i] + steps[i] * 4
        for j in range(0, 4):
            sequence.append(starts[i] + steps[i] * j)
        data_sequence.append(sequence)
        data_target.append(target)
    data_seq_tensor = torch.tensor(data_sequence, dtype=torch.float32)
    data_seq_tensor = (data_seq_tensor - data_seq_tensor.mean(0)) / data_seq_tensor.std(
        0
    )
    data_target_tensor = torch.tensor(data_target, dtype=torch.float32)
    data_target_tensor = (
        data_target_tensor - data_target_tensor.mean(0)
    ) / data_target_tensor.std(0)
    return data_seq_tensor, data_target_tensor


def train(model, n_epoch, loss_fn, optimizer, features, labels):
    for epoch in range(n_epoch):
        h = torch.zeros(100, 16)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features, h)
        loss = loss_fn(prediction, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(loss.item())


if __name__ == "__main__":
    model = RNNFromScratch(input_size=1, hidden_size=16, output_size=1)
    x, targets = fake_data_creator()
    x, targets = x.unsqueeze(-1), targets.unsqueeze(-1)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, 500, loss_fn, optimizer, x, targets)
