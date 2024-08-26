import math
from typing import List

import torch
from operator import attrgetter
from torch import nn


class Embedder(nn.Module):
    def __init__(self, n_entities, dim, max_length, use_position, dropout=0.2):
        super().__init__()
        self.node_embedding = nn.Embedding(n_entities+2, dim, padding_idx=0)
        if use_position:
            self.positional_embedding = nn.Embedding(max_length, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, items):
        mask = items != self.node_embedding.padding_idx
        embeddings = self.node_embedding(items) * mask.unsqueeze(-1)

        # Only use position if data has time and position therefore matters.
        if hasattr(self, 'positional_embedding'):
            embeddings += self.positional_embedding.weight.repeat(embeddings.size(0), 1, 1) * mask.unsqueeze(-1)

        embeddings = self.dropout(self.norm(embeddings))
        return embeddings, mask


class Transformer(nn.Module):
    def __init__(self, dim, num_heads, max_length, dropout=0.2, att_dropout=0.2):
        super().__init__()

        # Variables
        assert dim % num_heads == 0
        self._dim = dim
        self._num_heads = num_heads
        self._max_length = max_length

        # General
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Multi-head self-attention
        self.w_o = nn.Linear(dim, dim)
        self.w_q = nn.Parameter(torch.ones((dim, dim)))
        self.w_k = nn.Parameter(torch.ones((dim, dim)))
        self.w_v = nn.Parameter(torch.ones((dim, dim)))
        self.temperature = math.sqrt(dim / num_heads)

        # Position-wise feed-forwards network
        self.w_1 = nn.Linear(dim, 4*dim)
        self.w_2 = nn.Linear(4*dim, dim)
        self.activation = nn.GELU()

    def multi_head_self_attention(self, x, mask=None):
        # eq 1.2
        b, f, d = x.size()
        q, k, v = [(x @ w).reshape(b, f, self._num_heads, -1).transpose(1, 2)
                   for w in [self.w_q, self.w_k, self.w_v]]
        _, n, t, h = q.size()

        # eq 2 # bhsj x bhsj -> bhss
        # att_scores = torch.einsum('bnfh,bnth->bnft', q, k) / self.temperature
        att_scores = q @ k.transpose(-2, -1) / self.temperature

        if mask is not None:
            mask = mask.reshape(b, 1, 1, f)
            att_scores =att_scores.masked_fill(mask == 0, float('-10000'))

        att_probs = torch.softmax(att_scores, dim=-1)  # softmax over s

        context = torch.matmul(self.att_dropout(att_probs), v)
        context2 = context.transpose(1, 2).contiguous().view(b, f, d)  # bhsj -> bshj -> bsh

        # eq 1.1
        out = self.w_o(context2)
        return out

    def position_wise_fnn(self, x):
        return self.w_2(self.activation(self.w_1(x)))

    def forward(self, x, mask=None):
        # eq 1
        mh_att = self.multi_head_self_attention(x, mask)
        a_minus_1 = self.norm1(x + self.dropout(mh_att))

        # eq 3
        pw = self.position_wise_fnn(a_minus_1)
        h = self.norm2(a_minus_1 + self.dropout(pw))

        return h


class Prediction(nn.Module):
    def __init__(self, n_items, in_dim, out_dim, as_code=True):
        super().__init__()
        self.as_code = as_code
        self.w_p = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(n_items))
        self.activation = nn.GELU()

    def forward(self, x, item_embeddings, pred_idx=None):
        if pred_idx is not None:
            x = x[pred_idx]

        if self.as_code:
            return x @ item_embeddings.t() + self.bias
        else:
            # eq 7
            h = self.activation(self.w_p(x))
            p = torch.matmul(h, item_embeddings.T) + self.bias
            return p #torch.softmax(x, dim=-1)


class BERT4Rec(nn.Module):
    def __init__(self, n_entities, max_sequence_length, in_dim, num_layers, num_heads, dropout=0.2,
                 use_positional=False):
        super().__init__()
        self.embedding = Embedder(n_entities, in_dim, max_sequence_length, use_positional, dropout=dropout)

        self.transformers = nn.ModuleList([
            Transformer(in_dim, num_heads, max_sequence_length, dropout=dropout, att_dropout=dropout)
            for _ in range(num_layers)
        ])

        self.pred = Prediction(n_entities, in_dim, in_dim, as_code=False)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self.named_parameters():
            if 'bias' in k:
                nn.init.zeros_(v)
            else:
                nn.init.normal_(v, std=0.02)

    def embedder(self, items, pred_idx=None):
        x, masks = self.embedding(items)
        out = [x]
        for layer in self.transformers:
            x = layer(out[-1], masks)
            out.append(x)

        # Exclude padding and start/end tokens from node embedding and use in predictions
        pred = self.pred(out[-1], self.embedding.node_embedding.weight[1:-1], pred_idx)

        return pred
