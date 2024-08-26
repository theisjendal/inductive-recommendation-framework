import torch
from torch import nn


class BPR(nn.Module):
    def __init__(self, n_users, n_items, hidden_dim, user_bias=True, item_bias=True):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, hidden_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        self.item_emb = nn.Embedding(n_items, hidden_dim)
        nn.init.xavier_uniform_(self.item_emb.weight)

        if user_bias:
            self.user_bias = nn.Embedding(n_users, 1,)

        if item_bias:
            self.item_bias = nn.Embedding(n_items, 1)

        self.loss_fn = nn.LogSigmoid()

    def forward(self, users, items, rank_all=False):
        if rank_all:
            item_emb = self.item_emb(items).T

            pred = torch.matmul(self.user_emb(users), item_emb)

            if hasattr(self, 'item_bias'):
                pred += self.item_bias(items).T

            if hasattr(self, 'user_bias'):
                pred += self.user_bias(users)

        else:
            user_emb = self.user_emb(users)
            item_emb = self.item_emb(items)
            multi_neg = len(items) > len(users)
            if multi_neg:
                pred = torch.einsum('bd,bnd->bn', user_emb, item_emb.reshape(len(users), -1, item_emb.shape[-1]))
            else:
                pred = torch.mul(user_emb, item_emb).sum(axis=-1)

            if hasattr(self, 'item_bias'):
                i_bias = self.item_bias(items)
                if multi_neg:
                    pred += i_bias.reshape(pred.shape)
                else:
                    pred += i_bias.squeeze(-1)

            if hasattr(self, 'user_bias'):
                u_bias = self.user_bias(users)
                if multi_neg:
                    pred += u_bias
                else:
                    pred += u_bias.squeeze(-1)

        return pred

    def loss(self, users, items_i, items_j):
        ui = self.forward(users, items_i)
        uj = self.forward(users, items_j)

        if ui.shape == uj.shape:
            loss = ui - uj
        else:
            loss = (ui.unsqueeze(-1) - uj).flatten()

        return -self.loss_fn(loss).mean()


