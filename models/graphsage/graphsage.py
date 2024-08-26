import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv

from shared.graph_utility import dgl_homogeneous


class GraphSAGEDGL(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGEDGL, self).__init__()
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

    def forward(self, blocks, inputs):
        h = self.dropout(inputs)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class GraphSAGEPredictor(nn.Module):
    def __init__(self, train, meta, num_layers, aggregator, dim, num_samples, dropout, features, use_cuda):
        super().__init__()
        self.n_entities = len(meta.entities)
        n_users = len(meta.users)
        self.n_samples = num_samples
        self.use_cuda = use_cuda

        dgl_kg = dgl.add_self_loop(dgl_homogeneous(meta, meta.relations, num_nodes=self.n_entities))
        user_fn = lambda u: u + self.n_entities
        dgl_collab_kg = dgl_homogeneous(meta, meta.relations,
                                        [(user_fn(user.index), item, rating) for user in train
                                         for item, rating in user.ratings],
                                        num_nodes=n_users + self.n_entities)

        dgl_kg = dgl.to_bidirected(dgl_kg)
        dgl_collab_kg = dgl.to_bidirected(dgl_collab_kg)

        if aggregator == 'gcn':
            dgl_kg = dgl.add_self_loop(dgl_kg)
            dgl_collab_kg = dgl.add_self_loop(dgl_collab_kg)

        sampler = dgl.dataloading.MultiLayerNeighborSampler([num_samples for _ in range(num_layers)], replace=True)
        user_sampler = dgl.dataloading.MultiLayerNeighborSampler([num_samples], replace=True)
        self.collator = dgl.dataloading.NodeCollator(dgl_kg, torch.arange(self.n_entities), sampler)
        self.user_collator = dgl.dataloading.NodeCollator(dgl_collab_kg, torch.arange(self.n_entities + n_users),
                                                          user_sampler)

        # Feature is [train, val] or [train, test]
        o_features = []
        for feature_tmp in features:
            features_tmp = torch.tensor(feature_tmp, dtype=torch.float32)
            features_tmp = nn.Parameter(features_tmp, requires_grad=False)

            o_feature = nn.Embedding(*features_tmp.shape)
            o_feature.weight = features_tmp
            o_features.append(o_feature)

        self.features = nn.ModuleList(o_features)

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = nn.LogSigmoid()

        self._model = GraphSAGEDGL(features_tmp.shape[-1], dim, 64, num_layers-1, nn.ReLU(), dropout, aggregator)

    def forward(self, users, items, rank_all=False):
        users = users + self.n_entities

        # Get unique users and items and map to inverse them
        unique_users, inverse_users = torch.unique(users, return_inverse=True)
        unique_items, inverse_items = torch.unique(items, return_inverse=True)

        # Use item collater to get rated items in graph.
        rated_nodes, user_nodes, u_blocks = self.user_collator.collate(unique_users.cpu())

        # Above returns users nodes aswell, therefore remove all below user index.
        mask = rated_nodes < self.n_entities
        unique_rated, inverse_rated = torch.unique(rated_nodes[mask], return_inverse=True)

        unique_all, inverse_all = torch.unique(torch.cat([unique_items, unique_rated]), return_inverse=True)

        input_nodes, _, blocks = self.collator.collate(unique_all)

        if self.use_cuda:
            input_nodes = input_nodes.cuda()
            blocks = [b.to('cuda:0') for b in blocks]

        all_emb = self._model(blocks, self.features(input_nodes))[inverse_all]
        item_emb = all_emb[:len(unique_items)]
        rated_emb = all_emb[len(unique_items):]

        # Add zeroes to match with length of rated nodes as first nodes are user nodes and irrelevant
        rated_emb = F.pad(rated_emb, (0,0,len(rated_nodes)-torch.sum(mask),0), value=0.)

        # Get source nodes and change view to match number of samples.
        source, _ = u_blocks[0].edges(form='uv')
        source = source.view(-1, self.n_samples)[inverse_users].view(-1)

        if rank_all:
            # matrix cosine similarity.
            eps = 1e-8

            # Normalize and set min ensuring no zero division
            item_n = item_emb / torch.clamp(item_emb.norm(dim=1)[:, None], min=eps)
            rated_n = rated_emb / torch.clamp(rated_emb.norm(dim=1)[:, None], min=eps)

            # Get similarity between all rated embeddings and items.
            similarity = torch.mm(rated_n, item_n.transpose(0, 1))

            # Inverse mapping
            similarity = similarity[source][:, inverse_items]

            pred = similarity.view(len(users), self.n_samples, -1).sum(dim=1)
        else:
            # Repeat item embeddings to do matrix multiplication with user embeddings.
            item_emb = torch.repeat_interleave(item_emb[inverse_items], self.n_samples, dim=0)

            similarity = self.cosine_similarity(rated_emb[source], item_emb)

            pred = similarity.view(-1, self.n_samples).sum(dim=1)

        return pred

    def loss(self, users, items_i, items_j):
        pos_preds = self.forward(users, items_i)
        neg_preds = self.forward(users, items_j)

        loss = - self.criterion(pos_preds - neg_preds).sum()

        return loss


class GraphSAGEDGLPredictor(nn.Module):
    def __init__(self, graphs, n_entities, num_layers, aggregator, dim, num_samples, dropout, features, use_cuda):
        super().__init__()
        self.n_entities = n_entities
        self.n_samples = 10
        self.use_cuda = use_cuda

        (train_kg, test_kg), (train_cg, test_cg) = graphs

        dgl_kg = {k: dgl.to_bidirected(dgl_kg.to_simple(), copy_ndata=True) for k, dgl_kg in
                  [(True, train_kg), (False, test_kg)]}
        dgl_collab_g = {k: dgl.to_bidirected(dgl_cg.to_simple(), copy_ndata=True) for k, dgl_cg in [(True, train_cg), (False, test_cg)]}

        if aggregator == 'gcn':
            dgl_kg = {k: dgl.add_self_loop(kg) for k, kg in dgl_kg.items()}
            dgl_collab_g = {k: dgl.add_self_loop(g) for k, g in dgl_collab_g.items()}

        sampler = dgl.dataloading.MultiLayerNeighborSampler([num_samples for _ in range(num_layers)], replace=True)
        user_sampler = dgl.dataloading.MultiLayerNeighborSampler([num_samples], replace=True)
        self.collator = {k: dgl.dataloading.NodeCollator(kg, torch.arange(kg.number_of_nodes()), sampler)
                         for k, kg in dgl_kg.items()}
        self.user_collator = {k: dgl.dataloading.NodeCollator(g, torch.arange(g.number_of_nodes()), user_sampler)
                              for k, g in dgl_collab_g.items()}

        self.features = features
        self.latent_dim = features[0].shape[-1]

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = nn.LogSigmoid()

        self._model = GraphSAGEDGL(features[0].shape[-1], dim, 64, num_layers-1, nn.ReLU(), dropout, aggregator)

    def forward(self, users, items, rank_all=False):
        # Get unique users and items and map to inverse them
        unique_users, inverse_users = torch.unique(users, return_inverse=True)
        unique_items, inverse_items = torch.unique(items, return_inverse=True)

        # Use item collater to get rated items in graph.
        rated_nodes, _, u_blocks = self.user_collator[self.training].collate(unique_users.cpu())

        rated_nodes = rated_nodes.to(users.device)
        u_blocks = [block.to(users.device) for block in u_blocks]

        # Above returns users nodes aswell, therefore remove all below user index.
        mask = rated_nodes < self.n_entities
        unique_rated, inverse_rated = torch.unique(rated_nodes[mask], return_inverse=True)

        unique_all, inverse_all = torch.unique(torch.cat([unique_items, unique_rated]), return_inverse=True)

        input_nodes, _, blocks = self.collator[self.training].collate(unique_all.cpu())

        # Similar to above, mask features
        # features = torch.FloatTensor(np.array(self.features[input_nodes]))
        if self.training:
            features = self.features[0][input_nodes]
        else:
            features = self.features[1][input_nodes]

        if self.use_cuda:
            blocks = [b.to('cuda:0') for b in blocks]
            unique_items = unique_items.cuda()

        all_emb = self._model(blocks, features)[inverse_all]
        item_emb = all_emb[:len(unique_items)]
        rated_emb = all_emb[len(unique_items):]

        # Add zeroes to match with length of rated nodes as first nodes are user nodes and irrelevant
        rated_emb = F.pad(rated_emb, (0, 0, len(rated_nodes)-torch.sum(mask),0), value=0.)

        # Get source nodes and change view to match number of samples.
        source, _ = u_blocks[0].edges(form='uv')
        source = source.view(-1, self.n_samples)[inverse_users].view(-1)

        if rank_all:
            # matrix cosine similarity.
            eps = 1e-8

            # Normalize and set min ensuring no zero division
            item_n = item_emb / torch.clamp(item_emb.norm(dim=1)[:, None], min=eps)
            rated_n = rated_emb / torch.clamp(rated_emb.norm(dim=1)[:, None], min=eps)

            # Get similarity between all rated embeddings and items.
            similarity = torch.mm(rated_n, item_n.transpose(0, 1))

            # Inverse mapping
            similarity = similarity[source][:, inverse_items]

            pred = similarity.view(len(users), self.n_samples, -1).sum(dim=1)
        else:
            # Repeat item embeddings to do matrix multiplication with user embeddings.
            item_emb = torch.repeat_interleave(item_emb[inverse_items], self.n_samples, dim=0)

            similarity = self.cosine_similarity(rated_emb[source], item_emb)

            pred = similarity.view(-1, self.n_samples).sum(dim=1)

        return pred

    def loss(self, users, items_i, items_j):
        pos_preds = self.forward(users, items_i)
        neg_preds = self.forward(users, items_j)

        loss = - self.criterion(pos_preds - neg_preds).sum()

        return loss