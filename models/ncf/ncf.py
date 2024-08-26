import dgl
import torch
from torch import nn

from models.bpr.path_aggregator import RelationAggregator


class NCF(nn.Module):
    def __init__(self, in_dim, layer_dims, aggregator=False, *args):
        super().__init__()
        self.loss_fn = nn.LogSigmoid()

        if aggregator:
            self.aggregator = RelationAggregator(*args)
            in_dim = self.aggregator.out_dim

        self.mlp_layers = nn.ModuleList()
        in_dim *= 2  # Concatenate
        for out_dim in layer_dims + [1]:
            l = nn.Linear(in_dim, out_dim, bias=True)
            nn.init.xavier_uniform_(l.weight)
            self.mlp_layers.append(l)
            in_dim = out_dim

        self.activation = nn.ReLU()
        self.loss_fn = nn.LogSigmoid()

    def prediction(self, user_emb, item_emb):
        x = torch.cat([user_emb, item_emb], dim=-1)
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)

            if i + 1 < len(self.mlp_layers):
                x = self.activation(x)

        return x

    def forward(self, users, items, features):
        n_users = len(users)
        if hasattr(self, 'aggregator'):
            user_emb = self.aggregator(features[users])
            item_emb = self.aggregator(features[items])
            user_emb = user_emb[torch.repeat_interleave(torch.arange(n_users), len(items))]
            item_emb = item_emb[torch.arange(len(items)).repeat(n_users)]
        else:
            users = torch.repeat_interleave(users, len(items))
            items = items.repeat(n_users)
            user_emb, item_emb = features[users], features[items]

        x = self.prediction(user_emb, item_emb)
        x = x.reshape(n_users, -1)
        return x

    def graph_pred(self, g: dgl.DGLGraph, features):
        with g.local_scope():
            users, items = g.edges('uv')
            user_emb, item_emb = features[users], features[items]
            return self.prediction(user_emb, item_emb)

    def loss(self, pos_graph, neg_graph, features):
        if hasattr(self, 'aggregator'):
            features = self.aggregator(features)

        ui = self.graph_pred(pos_graph, features)
        uj = self.graph_pred(neg_graph, features)

        if ui.shape == uj.shape:
            loss = ui - uj
        else:
            loss = (ui.unsqueeze(-1) - uj).flatten()

        return -self.loss_fn(loss).mean(), ui > uj


