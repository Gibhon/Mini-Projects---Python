import torch
import torch.nn as nn


class LSTMFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev, c_prev):
        # x: [batch, input_size]
        # h_prev, c_prev: [batch, hidden_size

        combined = torch.cat([x, h_prev], dim=1)

        f_g = self.sigmoid(self.forget_gate(combined))
        i_g = self.sigmoid(self.input_gate(combined))
        o_g = self.sigmoid(self.output_gate(combined))
        c = self.tanh(self.candidate(combined))

        c_next = (f_g * c_prev) + (i_g * c)
        h_next = o_g * self.tanh(c_next)

        return c_next, h_next


class LSTMSequence(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMFromScratch(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            x_t = x[:, t, :]

            c, h = self.cell(x_t, h, c)

        output = self.fc(h)

        return output

        # TODO: loop over seq_len, slice x_t = x[:, t, :], update h and c

        # TODO: pass final h through self.fc, return output


if __name__ == "__main__":
    batch_size = 4
    seq_len = 7
    input_size = 10

    hidden_size = 20
    output_size = 3

    inputs = torch.randn([batch_size, seq_len, input_size])

    ltsm_model = LSTMSequence(input_size, hidden_size, output_size)
    output = ltsm_model(inputs)

    print(output.shape)
