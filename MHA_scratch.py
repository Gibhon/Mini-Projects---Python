import torch
import torch.nn as nn

seq_length = 50
d_model = 100
heads = 10

x = torch.randn(seq_length, d_model)


def multi_head_attention(x, num_heads, d_model):
    d_k = d_model // num_heads

    outputs = torch.tensor([])

    for i in range(num_heads):
        torch.manual_seed(i)
        query_transformer = torch.randn(d_model, d_k)
        key_transformer = torch.randn(d_model, d_k)
        value_transformer = torch.randn(d_model, d_k)

        query_tensor = torch.matmul(x, query_transformer)
        key_tensor = torch.matmul(x, key_transformer)
        value_tensor = torch.matmul(x, value_transformer)

        raw_weights = torch.matmul(query_tensor, key_tensor.T) / (d_k**0.5)
        weights = torch.softmax(raw_weights, dim=1)

        output_ph = torch.matmul(weights, value_tensor)

        outputs = torch.cat([outputs, output_ph], dim=-1)

    layer = nn.Linear(d_model, d_model)

    return layer(outputs)


output = multi_head_attention(x, heads, d_model)
print(output.shape)
