import torch
import pytorch_lightning as pl

from torch import nn, Tensor
from local_attention import LocalAttention
from npz_trajectory import NpzTrajectory


class EgoTrajectoryEncoder(nn.Module):
    def __init__(
        self,
        max_dist=50.0,
        # num_timesteps=11,
        num_layers=6,
        dim_model=128,
        num_heads=8,
        dim_feedforward=512,
        dropout=0.1,
        dim_output=256,
    ):
        super().__init__()
        self.to_dim_model = nn.Linear(
            in_features=3,  # 2 pos, 1 temp
            out_features=dim_model,
        )
        self.max_dist = max_dist
        # self.num_timesteps = num_timesteps

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_output)

    def forward(self, pos_src_tokens):
        pos_src_tokens /= self.max_dist
        batch_size = pos_src_tokens.size(0)
        time_encoding = torch.arange(
            0, pos_src_tokens.shape[1], device=pos_src_tokens.device
        ) / (pos_src_tokens.shape[1] - 1)
        time_encoding = time_encoding.expand(batch_size, -1)[:, :, None]

        src = torch.cat(
            (
                pos_src_tokens,
                time_encoding,
            ),
            dim=2,
        )
        src = self.to_dim_model(src)

        for layer in self.layers:
            src = layer(src, src, src)

        return self.linear(src)


# class EgoTrajectoryEncoder(nn.Module):
#     def __init__(
#         self,
#         dim_semantic_embedding=4,
#         max_dist=50.0,
#         num_timesteps=11,
#         num_layers=6,
#         dim_model=128,
#         num_heads=8,
#         dim_feedforward=512,
#         dropout=0.1,
#         dim_output=256,
#     ):
#         super().__init__()
#         self.to_dim_model = nn.Linear(
#             in_features=dim_semantic_embedding + 3,  # 2 pos, 1 temp
#             out_features=dim_model,
#         )
#         self.semantic_embedding = nn.Embedding(
#             num_embeddings=6,  # Classes static + dynamic
#             embedding_dim=dim_semantic_embedding,
#         )
#         self.max_dist = max_dist
#         self.num_timesteps = num_timesteps

#         self.layers = nn.ModuleList(
#             [
#                 TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.linear = nn.Linear(dim_model, dim_output)

#     def forward(self, idxs_semantic_embedding, pos_src_tokens):
#         pos_src_tokens /= self.max_dist
#         batch_size = idxs_semantic_embedding.size(dim=0)
#         time_encoding = torch.arange(0, self.num_timesteps) / (self.num_timesteps - 1)
#         time_encoding = time_encoding.expand(batch_size, -1)[:, :, None]
#         time_encoding = time_encoding.to("cuda")

#         src = torch.concat(
#             (
#                 self.semantic_embedding(idxs_semantic_embedding),
#                 pos_src_tokens,
#                 time_encoding,
#             ),
#             dim=2,
#         )
#         src = self.to_dim_model(src)

#         for layer in self.layers:
#             src = layer(src, src, src)

#         return self.linear(src)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.attn = Residual(
            nn.MultiheadAttention(
                embed_dim=dim_model,
                num_heads=num_heads,
                batch_first=True,
            ),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src_q, src_k, src_v):
        attn_out = self.attn(src_q, src_k, src_v, need_weights=False)
        return self.feed_forward(attn_out)


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor, **kwargs: dict) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        output_sublayer = self.sublayer(*tensors, **kwargs)

        # nn.MultiheadAttention always returns a tuple (out, attn_weights or None)
        if isinstance(output_sublayer, tuple):
            output_sublayer = output_sublayer[0]

        return self.norm(tensors[0] + self.dropout(output_sublayer))


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


# Initialize the encoder
encoder = EgoTrajectoryEncoder()

npz = NpzTrajectory(
    "/mrtstorage/datasets/tmp/waymo_open_motion_processed/train-2e6/vehicle_b_78219_00001_4426517365.npz"
)
print(npz.coordinates.shape)
print(type(npz.coordinates.shape))

# Create a dummy trajectory
# Assuming batch_size = 1 and num_timesteps = 11
batch_size = 1
data = npz.coordinates.to_numpy()
print(data)
pos_src_tokens = torch.Tensor(data.reshape(batch_size, data.shape[0], data.shape[1]))

# trajectory_length = 100

# trajectory = (
#     torch.linspace(0, 1, steps=trajectory_length).repeat(batch_size, 1).unsqueeze(-1)
# )

# print(trajectory.shape)  # Should print [batch_size, trajectory_length, 1]

# Concatenate trajectory with itself along the last dimension
# trajectory = torch.cat((trajectory, trajectory), dim=-1)  # Same values for x and y
# print("Done")
# print(trajectory.shape)

# Concatenate trajectory with time steps
# pos_src_tokens = trajectory

# # Process the trajectory through the encoder
# # print("Encoder Call reached")
output = encoder(pos_src_tokens)

# Output
print("Output shape:", output.shape)
print(output)
