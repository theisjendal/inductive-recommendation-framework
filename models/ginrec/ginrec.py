from typing import List

import dgl
import torch
import torch.nn as nn
from dgl.heterograph import DGLBlock

from models.ginrec.autoencoder import Autoencoder
from models.ginrec.ginrec_conv import GInRecConv
from shared.enums import FeatureEnum


class GInRec(nn.Module):
    def __init__(self, n_relations, entity_dim, user_fn, autoencoder_layer_dims,
                 dropouts, dimensions, n_edges, gate_type='concat', aggregator='gcn',
                 attention='gatv2', attention_dim=32, device='cpu', relations=True, activation=False,
                 tanh_range=False, use_out_layer=False, use_initial_embedding=True):
        super(GInRec, self).__init__()

        self.use_initial_embedding = use_initial_embedding
        self.user_fn = user_fn
        self.n_entities = user_fn(0)
        self.n_layers = len(dimensions)

        # if light gcn keep dimensions throughout
        self.layer_dims = dimensions if aggregator != 'lightgcn' else [autoencoder_layer_dims[-1]]*self.n_layers
        self.device = device
        self.autoencoder = Autoencoder(entity_dim, autoencoder_layer_dims, use_activation=activation,
                                       tanh_range=tanh_range)
        self.gate_type = gate_type

        # Entity dimensions are reduced after the autoencoder.
        input_dim = autoencoder_layer_dims[-1]

        if gate_type not in ['concat', 'inner_product', None]:
            raise ValueError('Invalid gate type.')

        if attention is not None:
            self.alpha_r = nn.Parameter(torch.Tensor(self.n_layers, n_relations, attention_dim))
        else:
            self.alpha_r = None

        self.W_r = nn.ParameterList() if gate_type is not None else None
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for output_dim, dropout in zip(self.layer_dims, dropouts):
            if gate_type == 'concat':
                r_dim = input_dim * 2
            else:
                r_dim = input_dim

            if self.W_r is not None:
                W_r = nn.Parameter(torch.Tensor(n_relations, r_dim, input_dim))
                nn.init.xavier_uniform_(W_r, gain=nn.init.calculate_gain('relu'))
                self.W_r.append(W_r)

            # Add layers where the last layer skips activation
            self.layers.append(GInRecConv(input_dim, output_dim, gate_type=gate_type, aggregator=aggregator,
                                          attention=attention, attention_dim=attention_dim, relations=relations))

            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.pred_activation = nn.ReLU()

        self.use_out_layer = use_out_layer
        if use_out_layer:
            in_dim = sum([d for d in self.layer_dims])
            if self.use_initial_embedding:
                in_dim += self.layer_dims[0]
            self.W_out = nn.Linear(in_dim, self.layer_dims[-1])

        sigmoid = torch.nn.LogSigmoid()
        self.loss_fn = lambda pos, neg: - torch.mean(sigmoid(pos - neg))
        self.embeddings = None
        self.edge_weights = nn.Parameter(torch.zeros(n_edges, 1, dtype=torch.float32), requires_grad=False)

        # self._initialize_weights()

    def _initialize_weights(self):
        for name, parameter in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(parameter, 0.0)
            else:
                nn.init.xavier_uniform_(parameter)

    def embedder(self, blocks: List[DGLBlock], x: torch.FloatTensor):
        dstnodes = blocks[-1].dstnodes()
        embeddings = []

        if self.use_initial_embedding:
            embeddings.append(x[dstnodes])

        for l, layer in enumerate(self.layers):
            x = self.dropouts[l](x)
            x = layer(blocks[l], x, self.W_r[l] if self.gate_type is not None else None,
                      self.alpha_r[l] if self.alpha_r is not None else None)

            # Only interested in item nodes for disentanglement, i.e., not h.
            embeddings.append(x[dstnodes])

        embeddings = torch.cat(embeddings, dim=-1)

        if self.use_out_layer:
            embeddings['h'] = self.W_out(embeddings['h'])

        return embeddings

    def predict(self, g: dgl.DGLGraph, embeddings):
        with g.local_scope():
            users, items = g.edges()
            user_emb, item_emb = embeddings[users], embeddings[items]
            x = (user_emb * item_emb).sum(dim=1)

            return x

    def _get_dataloader(self, g, batch_size):
        sampler = dgl.dataloading.MultiLayerNeighborSampler([-1], prefetch_edge_feats=['type'])
        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_edge_feats=['type'])
        dataloader = dgl.dataloading.NodeDataLoader(
            g, torch.arange(g.number_of_nodes()), sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False, device=self.device)
        return dataloader

    def inference(self, g: dgl.DGLGraph, embeddings, batch_size=128):
        embeddings = self.autoencoder.propagate(embeddings, 'encode')
        if self.use_initial_embedding:
            all_emb = [embeddings]
        else:
            all_emb = []

        all_gate_weights = []
        for l, layer in enumerate(self.layers):
            next_embeddings = torch.zeros((g.number_of_nodes(), layer.output_dim), device=embeddings.device)

            dataloader = self._get_dataloader(g, batch_size)

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]
                nodes = block.srcdata[dgl.NID]
                w_r = self.W_r[l] if self.W_r is not None else None
                alpha = self.alpha_r[l] if self.alpha_r is not None else None
                next_embeddings[output_nodes] = layer(block, embeddings[nodes], w_r, alpha)

            embeddings = next_embeddings
            all_emb.append(embeddings)

        self.embeddings = torch.cat(all_emb, dim=-1)
        if all_gate_weights:
            self.edge_weights.data = torch.mean(torch.stack(all_gate_weights), dim=0)

    def forward(self, users, items, rank_all=False, apply_user_fn=True):
        if apply_user_fn:
            users = self.user_fn(users)

        users, items = users.to(self.device), items.to(self.device)

        user_embs = self.embeddings[users]
        item_embs = self.embeddings[items]

        if rank_all:
            predictions = torch.matmul(user_embs, item_embs.T)
        else:
            predictions = (user_embs * item_embs).sum(dim=1)

        return predictions

    def loss_ae(self, embeddings, ntypes=None):
        if isinstance(embeddings, dict):
            encoded = []
            ae_loss = []
            for enum in sorted(FeatureEnum):
                enum = enum.name
                if enum in embeddings:

                    emb = embeddings[enum]
                    enc, decoded = self.autoencoder[enum](emb, ntypes)
                    ae_loss.append(self.autoencoder[enum].loss(emb, decoded, enc, ntypes).mean(dim=-1))
                    encoded.append(enc)

            encoded = torch.cat(encoded, dim=0)
            ae_loss = torch.cat(ae_loss)

        else:
            encoded, decoded = self.autoencoder(embeddings)
            ae_loss = self.autoencoder.loss(embeddings, decoded)

        # encoded = encoded[input_nodes]
        ae_loss = ae_loss.mean()

        return encoded, ae_loss

    def loss(self, pos_graph, neg_graph, blocks, embeddings):
        graph_embeddings = self.embedder(blocks, embeddings)
        pos_preds, neg_preds = self.predict(pos_graph, graph_embeddings), self.predict(neg_graph, graph_embeddings)

        pos_preds, neg_preds = pos_preds.unsqueeze(-1), neg_preds.view(pos_preds.shape[0], -1)
        cf_loss = self.loss_fn(pos_preds, neg_preds)

        l2_loss = self.l2_loss(pos_graph, neg_graph, graph_embeddings)

        return cf_loss, l2_loss, pos_preds > neg_preds

    def l2_loss(self, pos_graph, neg_graph, graph_embeddings):
        u, i = pos_graph.edges()
        u_emb, i_emb = graph_embeddings[u], graph_embeddings[i]
        _, j = neg_graph.edges()
        j_emb = graph_embeddings[j]

        l2_loss = (1/2)*(u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2) + j_emb.norm(2).pow(2)) / float(len(u))

        return l2_loss

    def get_edge_weights(self, g, l, embeddings):
        dataloader = self._get_dataloader(g, -1, 256, 0)
        gate_weights = torch.zeros((g.number_of_edges(), 1), device=embeddings.device)
        for input_nodes, output_nodes, blocks in dataloader:
            block = blocks[0]
            nodes = block.srcdata[dgl.NID]
            gate_weights[block.edata[dgl.EID]] = self.layers[l].get_gate_weight(block, embeddings[nodes], self.W_r[l]
            if self.W_r is not None else None)

        return gate_weights.squeeze(-1)