import torch

seq_len = 5
d_model = 8

x = torch.randn(seq_len, d_model)
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)


def attention(x, query_transformer, key_transformer, value_transformer):
    query_tensor = torch.matmul(x, query_transformer)
    key_tensor = torch.matmul(x, key_transformer)
    value_tensor = torch.matmul(x, value_transformer)

    raw_weights = torch.matmul(query_tensor, key_tensor.T) / (d_model**0.5)
    weights = torch.softmax(raw_weights, dim=1)

    output = torch.matmul(weights, value_tensor)
    return output


out = attention(x, W_q, W_k, W_v)
print(out.shape)


print(torch.softmax(torch.tensor([1.0, 2.0, 1.0]), dim=0))
print(torch.softmax(torch.tensor([10.0, 20.0, 10.0]), dim=0))
