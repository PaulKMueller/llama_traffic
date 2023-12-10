import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import torch
from torch import nn, Tensor


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


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


class EgoTrajectoryEncoder(nn.Module):
    def __init__(
        self,
        dim_semantic_embedding=4,
        max_dist=50.0,
        num_timesteps=11,  # 80?
        num_layers=6,
        dim_model=128,
        num_heads=8,
        dim_feedforward=512,
        dropout=0.1,
        dim_output=256,
    ):
        super().__init__()
        self.to_dim_model = nn.Linear(
            in_features=dim_semantic_embedding + 3,  # 2 pos, 1 temp
            out_features=dim_model,
        )
        self.semantic_embedding = nn.Embedding(
            num_embeddings=6,  # Classes static + dynamic
            embedding_dim=dim_semantic_embedding,
        )
        self.max_dist = max_dist
        self.num_timesteps = num_timesteps

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_output)

    def forward(self, idxs_semantic_embedding, pos_src_tokens):
        pos_src_tokens /= self.max_dist
        batch_size = idxs_semantic_embedding.size(dim=0)
        time_encoding = torch.arange(0, self.num_timesteps) / (self.num_timesteps - 1)
        time_encoding = time_encoding.expand(batch_size, -1)[:, :, None]
        time_encoding = time_encoding.to("cuda")

        src = torch.concat(
            (
                self.semantic_embedding(idxs_semantic_embedding),
                pos_src_tokens,
                time_encoding,
            ),
            dim=2,
        )
        src = self.to_dim_model(src)

        for layer in self.layers:
            src = layer(src, src, src)

        return self.linear(src)
