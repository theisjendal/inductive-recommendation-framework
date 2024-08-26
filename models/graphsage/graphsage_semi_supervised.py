import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv

from shared.graph_utility import dgl_homogeneous


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 features):
        super(GraphSAGE, self).__init__()
        self.features = features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def embedder(self, blocks, x):
        h = self.dropout(x)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def predict(self, g: dgl.DGLGraph, x):
        pass

    def _get_features(self, input_nodes):
        pass


    def loss(self, input_nodes, pos_graph, neg_graph, blocks):
        x = self._get_features(input_nodes)
        x = self.embedder(blocks, x)
        pos, neg = self.predict(pos_graph, x), self.predict(neg_graph, x)



