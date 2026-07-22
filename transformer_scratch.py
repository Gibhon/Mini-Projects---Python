import torch
import torch.nn as nn

# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
#         )

#     def forward(self, x):
#         return self.network(x)


# class LayerNormScratch(nn.Module):
#     def __init__(self, d_model, eps=1e-5):
#         super().__init__()
#         self.gammas = nn.Parameter(
#             torch.ones(
#                 d_model,
#             )
#         )
#         self.betas = nn.Parameter(
#             torch.zeros(
#                 d_model,
#             )
#         )
#         self.eps = eps

#     def forward(self, x):
#         mean_hor = x.mean(dim=-1, keepdim=True)
#         var_hor = x.var(dim=-1, keepdim=True, unbiased=False)

#         normalized_x = (x - mean_hor) / torch.sqrt(var_hor + self.eps)

#         return (self.gammas * normalized_x) + self.betas


# class MultiHeadAttention(nn.Module):
#     def __init__(self, num_heads, d_model):
#         super().__init__()
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.d_model = d_model

#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.output_layer = nn.Linear(d_model, d_model)

#     def forward(self, x):
#         seq_len = x.shape[0]

#         q = (
#             self.q_proj(x).view(seq_len, self.num_heads, self.d_k).permute(1, 0, 2)
#         )  # (S, D) @ (D, D) => (S, D) =>(View) (S, H, N) => (H, S, N)
#         k = (
#             self.k_proj(x).view(seq_len, self.num_heads, self.d_k).permute(1, 0, 2)
#         )  # (S, D) @ (D, D) => (S, D) =>(View) (S, H, N) => (H, S, N)
#         v = (
#             self.v_proj(x).view(seq_len, self.num_heads, self.d_k).permute(1, 0, 2)
#         )  # (S, D) @ (D, D) => (S, D) =>(View) (S, H, N) => (H, S, N)

#         scores = torch.matmul(q, k.transpose(-2, -1)) / (
#             self.d_k**0.5
#         )  # (H, S, N) @ (H, N, S)
#         softmax_weights = torch.softmax(scores, dim=-1)

#         context = torch.matmul(
#             softmax_weights, v
#         )  # (H, S, S) @ (H, S, N) ==> (H, S, N)
#         context = context.permute(1, 0, 2).contiguous()  # (S, H, N)

#         combined_heads = context.view(seq_len, self.d_model)  # (S, D)

#         return self.output_layer(combined_heads)  # (S, D) @ (D, D)


# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff):
#         # your submodules: MHA, FeedForward, 2x LayerNorm
#         super().__init__()
#         self.d_model = d_model
#         self.d_ff = d_ff
#         self.num_heads = num_heads

#         self.feed_forward = FeedForward(d_model, d_ff)
#         self.layernorm1 = LayerNormScratch(d_model=d_model)
#         self.layernorm2 = LayerNormScratch(d_model=d_model)
#         self.MHA = MultiHeadAttention(num_heads, d_model)

#     def forward(self, x):
#         x = x + self.MHA(x)
#         x = self.layernorm1(x)
#         x = x + self.feed_forward(x)
#         x = self.layernorm2(x)
#         return x


# def positional_encoding(seq_len, d_model) -> torch.Tensor:
#     pe = torch.zeros((seq_len, d_model))
#     position = torch.arange(seq_len).unsqueeze(-1)

#     div_term = 10000 ** (torch.arange(0, d_model, 2) / d_model)

#     pe[:, 0::2] = torch.sin(position / div_term)

#     pe[:, 1::2] = torch.cos(position / div_term)

#     return pe


# seq_len, d_model, num_heads, d_ff = 5, 8, 2, 32
# x = torch.randn(seq_len, d_model)
x = torch.randn(5, 8)
# block = TransformerEncoderBlock(d_model, num_heads, d_ff)
# out = block(x)
# print(out.shape)


encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=32)

print(encoder_layer(x).shape)
