import torch
import pytorch_lightning as pl

from torch import nn, Tensor
from local_attention import LocalAttention


class REDEncoder(pl.LightningModule):
    """Road Environment Description (RED) Encoder"""

    def __init__(
        self,
        size_encoder_vocab: int = 11,
        dim_encoder_semantic_embedding: int = 4,
        num_encoder_layers: int = 6,
        size_decoder_vocab: int = 100,
        num_decoder_layers: int = 6,
        dim_model: int = 512,
        dim_heads_encoder: int = 64,
        dim_attn_window_encoder: int = 64,
        num_heads_decoder: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_dist: float = 50.0,
        z_dim: int = 512,
        batch_size: int = 8,
        max_train_epochs: int = 200,
        learning_rate=1e-4,
        lambda_coeff=5e-3,
    ):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_encoder_vocab,
            embedding_dim=dim_encoder_semantic_embedding,
            padding_idx=-1,  # For [pad] token
        ).to(device)
        self.to_dim_model = nn.Linear(
            in_features=dim_encoder_semantic_embedding + 2,  # For position as (x, y)
            out_features=dim_model,
        )
        self.max_dist = max_dist
        self.encoder = LocalTransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            dim_heads=dim_heads_encoder,
            dim_attn_window=dim_attn_window_encoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.range_decoder_embedding = torch.arange(size_decoder_vocab).expand(
            batch_size, size_decoder_vocab
        )
        self.decoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_decoder_vocab,
            embedding_dim=dim_model - 10,  # For learned pos. embedding
        )
        self.decoder_pos_embedding = nn.Embedding(
            num_embeddings=size_decoder_vocab,
            embedding_dim=10,
        )

        self.decoder = ParallelTransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads_decoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.projection_head = nn.Sequential(
            nn.Linear(
                in_features=size_decoder_vocab * 2, out_features=4096
            ),  # Mean, var per token
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=z_dim),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = BarlowTwinsLoss(
            batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim
        ).to(device)
        self.max_epochs = max_train_epochs
        self.learning_rate = learning_rate

    def forward(
        self, idxs_src_tokens: Tensor, pos_src_tokens: Tensor, src_mask: Tensor
    ) -> Tensor:
        pos_src_tokens /= self.max_dist
        src = torch.concat(
            (self.encoder_semantic_embedding(idxs_src_tokens), pos_src_tokens), dim=2
        )  # Concat in feature dim
        src = self.to_dim_model(src)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.range_decoder_embedding = self.range_decoder_embedding.to(device)
        tgt = torch.concat(
            (
                self.decoder_semantic_embedding(self.range_decoder_embedding),
                self.decoder_pos_embedding(self.range_decoder_embedding),
            ),
            dim=2,
        )

        return self.decoder(tgt, self.encoder(src, src_mask), src_mask)

    def shared_step(self, batch):
        road_env_tokens_a = self.forward(
            idxs_src_tokens=batch["sample_a"]["idx_src_tokens"],
            pos_src_tokens=batch["sample_a"]["pos_src_tokens"],
            src_mask=batch["src_attn_mask"],
        )
        road_env_tokens_b = self.forward(
            idxs_src_tokens=batch["sample_b"]["idx_src_tokens"],
            pos_src_tokens=batch["sample_b"]["pos_src_tokens"],
            src_mask=batch["src_attn_mask"],
        )
        z_a = self.projection_head(
            torch.concat(
                (road_env_tokens_a.mean(dim=2), road_env_tokens_a.var(dim=2)), dim=1
            )
        )
        z_b = self.projection_head(
            torch.concat(
                (road_env_tokens_b.mean(dim=2), road_env_tokens_b.var(dim=2)), dim=1
            )
        )
        return self.loss_fn(z_a, z_b)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.max_epochs,
                    eta_min=1e-6,
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }


class ParallelTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = Residual(
            nn.MultiheadAttention(
                embed_dim=dim_model,
                num_heads=num_heads,
                batch_first=True,
            ),
            dimension=dim_model,
            dropout=dropout,
        )
        self.cross_attn = Residual(
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

    def forward(self, tgt: Tensor, memory: Tensor, mem_mask: Tensor) -> Tensor:
        tgt = self.self_attn(tgt, tgt, tgt, need_weights=False)
        batch_size, tgt_len = tgt.size(dim=0), tgt.size(dim=1)
        mem_mask = mem_mask[:, None, :].expand(batch_size, tgt_len, -1)
        mem_mask = mem_mask.repeat(1, self.num_heads, 1)
        mem_mask = mem_mask.view(batch_size * self.num_heads, tgt_len, -1)

        tgt = self.cross_attn(
            tgt, memory, memory, attn_mask=mem_mask, need_weights=False
        )

        return self.feed_forward(tgt)


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


class ParallelTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        add_pos_encoding: bool = False,
    ):
        super().__init__()
        self.add_pos_encoding = add_pos_encoding

        self.layers = nn.ModuleList(
            [
                ParallelTransformerDecoderLayer(
                    dim_model, num_heads, dim_feedforward, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor, mem_mask: Tensor) -> Tensor:
        if self.add_pos_encoding:
            seq_len, dimension = tgt.size(1), tgt.size(2)
            tgt += position_encoding(seq_len, dimension)

        for layer in self.layers:
            tgt = layer(tgt, memory, mem_mask)

        return self.linear(tgt)


def position_encoding(
    seq_len: int,
    dim_model: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim / dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


class LocalTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        dim_heads: int = 64,
        dim_attn_window: int = 64,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = Residual(
            LocalMultiheadAttention(
                dim_in=dim_model,
                dim_q=dim_model,
                dim_k=dim_model,
                dim_heads=dim_heads,
                dim_attn_window=dim_attn_window,
            ),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        src = self.attention(src, src, src, mask)
        return self.feed_forward(src)


class LocalTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        dim_heads: int = 64,
        dim_attn_window: int = 64,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        add_pos_encoding: bool = False,
    ):
        super().__init__()
        self.add_pos_encoding = add_pos_encoding
        self.layers = nn.ModuleList(
            [
                LocalTransformerEncoderLayer(
                    dim_model, dim_heads, dim_attn_window, dim_feedforward, dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        if self.add_pos_encoding:
            seq_len, dimension = src.size(1), src.size(2)
            src += position_encoding(seq_len, dimension)

        for layer in self.layers:
            src = layer(src, mask)

        return src


class LocalMultiheadAttention(nn.Module):
    def __init__(
        self, dim_in: int, dim_q: int, dim_k: int, dim_heads: int, dim_attn_window: int
    ):
        super().__init__()
        self.to_q = nn.Linear(dim_in, dim_q)
        self.to_k = nn.Linear(dim_in, dim_k)
        self.to_v = nn.Linear(dim_in, dim_k)
        self.attn = LocalAttention(
            dim=dim_heads,
            window_size=dim_attn_window,
            autopad=True,
            use_rotary_pos_emb=False,
        )

    def forward(self, queries, keys, values, mask):
        q = self.to_q(queries)
        k = self.to_k(keys)
        v = self.to_v(values)

        return self.attn(q, k, v, mask=mask)


class BarlowTwinsLoss(nn.Module):
    """Src: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/barlow-twins.html"""

    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming these parameters for the dummy data
batch_size = 8  # as defined in REDEncoder
seq_len = 10  # arbitrary sequence length
size_encoder_vocab = 11  # as defined in REDEncoder
dim_pos = 2  # for (x, y) coordinates

# Create dummy data
idxs_src_tokens = torch.randint(size_encoder_vocab, (batch_size, seq_len))
pos_src_tokens = (
    torch.rand(batch_size, seq_len, dim_pos) * 100
)  # assuming positional values
src_mask = torch.ones(batch_size, seq_len)  # assuming all tokens are valid

# Normalize positional information as per the REDEncoder forward method
pos_src_tokens /= 50.0  # Using the max_dist value from the REDEncoder

idxs_src_tokens = idxs_src_tokens.to(device)
pos_src_tokens = pos_src_tokens.to(device)
src_mask = src_mask.bool().to(device)

# Create an instance of the model
red_encoder = REDEncoder().to(device)

print(idxs_src_tokens.device)
print(pos_src_tokens.device)
print(src_mask.device)
print(red_encoder.device)

# Call the encoder with dummy data
output = red_encoder(idxs_src_tokens, pos_src_tokens, src_mask)

# Output shape will depend on the model's internal configurations
print("Output shape:", output.shape)
