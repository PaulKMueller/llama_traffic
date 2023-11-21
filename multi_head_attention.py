# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# import math
# import copy

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
        
#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
#         self.W_o = nn.Linear(d_model, d_model)
        
#     def scaled_dot_product_attention(self, Q, K, V, mask=None):
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
#         attn_probs = torch.softmax(attn_scores, dim=-1)
#         output = torch.matmul(attn_probs, V)
#         return output
        
#     def split_heads(self, x):
#         batch_size, seq_length, d_model = x.size()
#         return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
#     def combine_heads(self, x):
#         batch_size, _, seq_length, d_k = x.size()
#         return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
#     def forward(self, Q, K, V, mask=None):
#         Q = self.split_heads(self.W_q(Q))
#         K = self.split_heads(self.W_k(K))
#         V = self.split_heads(self.W_v(V))
        
#         attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
#         output = self.W_o(self.combine_heads(attn_output))
#         return output


# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super(PositionWiseFeedForward, self).__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.fc2(self.relu(self.fc1(x)))
    

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super(PositionalEncoding, self).__init__()
        
#         pe = torch.zeros(max_seq_length, d_model)
#         position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         self.register_buffer('pe', pe.unsqueeze(0))
        
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]
    

# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, mask):
#         attn_output = self.self_attn(x, x, x, mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + self.dropout(ff_output))
#         return x
    

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout):
#         super(DecoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.cross_attn = MultiHeadAttention(d_model, num_heads)
#         self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, enc_output, src_mask, tgt_mask):
#         attn_output = self.self_attn(x, x, x, tgt_mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
#         x = self.norm2(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.norm3(x + self.dropout(ff_output))
#         return x
    

# class Transformer(nn.Module):
#     def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
#         super(Transformer, self).__init__()
#         self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
#         self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
#         self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

#         self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
#         self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

#         self.fc = nn.Linear(d_model, tgt_vocab_size)
#         self.dropout = nn.Dropout(dropout)

#     def generate_mask(self, src, tgt):
#         src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
#         tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
#         seq_length = tgt.size(1)
#         nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
#         tgt_mask = tgt_mask & nopeak_mask
#         return src_mask, tgt_mask

#     def forward(self, src, tgt):
#         src_mask, tgt_mask = self.generate_mask(src, tgt)
#         src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
#         tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

#         enc_output = src_embedded
#         for enc_layer in self.encoder_layers:
#             enc_output = enc_layer(enc_output, src_mask)

#         dec_output = tgt_embedded
#         for dec_layer in self.decoder_layers:
#             dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

#         output = self.fc(dec_output)
#         return outputf


from typing import Optional, List
import torch
from torch import nn

import numpy as np
import math


def get_positional_encoding(sequence):
    """
    Get the positional encoding for a given sequence.

    :param sequence: The sequence to get the positional encoding for.
    :return: The positional encoding for the given sequence.
    """
    sequence_length, dimensions = sequence.shape
    pos_encoding = np.zeros((sequence_length, dimensions))

    for pos in range(sequence_length):
        for i in range(dimensions):
            if i % 2 == 0:
                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / dimensions)))
            else:
                pos_encoding[pos, i] = math.cos(pos / (10000 ** ((2 * i) / dimensions)))
    
    return pos_encoding + sequence


class PrepareForMultiHeadAttention(nn.Module):


    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()

        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        # d_k is the number of dimensions in vectors in each head
        self.d_k = d_k

    
    def forward(self, x: torch.Tensor):
        # Input has shape [seq_len, batch_size, d_model] or [batch_size, d_model]
        head_shape = x.shape[:-1]

        x = self.linear()
        # view is a PyTorch function similar to numpy.reshape()
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape [seq_len, batch_size, heads, d_k] or [batch_size, heads, d_model]

        return x
    

class MultiHeadAttention(nn.Module):


    def __init__(self, heads:int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        # heads is the number of heads.
        # d_model is the number of features in the query , key and value vectors.

        super.__init__()
        self.d_k = d_model
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        # Product of query and key matrix in Einstein Notation summation
        return torch.einsum('ibhd,jbhd->ijbh', query, key)
    

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        # mask has shape [seq_len_q, seq_len_k, batch_size]

        # Unsqueeze(-1) adds a dimension to the Tensor, e.g. [A, B] becomes [A, B, 1]
        mask = mask.unsqueeze(-1)

        return mask

    
    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)


        # Preparing the different Tensors
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Computing similarities
        scores = self.get_scores(query, key)

        # Scaling similarities
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)

        attn = self.dropout(attn)

        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # detach creates a new tensor that is disconnected from the computation graph
        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)

