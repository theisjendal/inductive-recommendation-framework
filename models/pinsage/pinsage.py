from typing import List

import dgl
import torch
from dgl import function as fn
from dgl.dataloading import DataLoader
from torch import nn

from models.shared.dgl_dataloader import PinSAGESampler, PinSAGESampler2


class PinSAGELayer(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.q = nn.Linear(in_dim, in_dim)
		self.w = nn.Linear(2*in_dim, out_dim)
		self.activation = nn.ReLU()

		# Init
		gain = nn.init.calculate_gain('relu')
		nn.init.xavier_uniform_(self.q.weight, gain=gain)
		nn.init.xavier_uniform_(self.w.weight, gain=gain)
		nn.init.constant_(self.q.bias, 0)
		nn.init.constant_(self.w.bias, 0)

	def forward(self, g: dgl.DGLGraph, features, ntype='_N'):
		with g.local_scope():
			feat_src = feat_dst = features[ntype]

			if g.is_block:
				feat_dst = feat_dst[:g.number_of_dst_nodes(ntype)]

			g.srcnodes[ntype].data['h'], g.dstnodes[ntype].data['h'] = feat_src, feat_dst

			# Algorithm 1 line 1
			g.srcnodes[ntype].data['h'] = self.activation(self.q(g.srcnodes[ntype].data['h']))  # Linear layer
			g.update_all(fn.u_mul_e('h', 'alpha', 'm'), fn.sum('m', 'h_n'))  # Gamma function

			# Algorithm 1 line 2
			h_n = self.activation(self.w(torch.cat([g.dstdata['h'], g.dstdata['h_n']], dim=-1)))

			# Algorithm 1 line 3
			h_n = h_n / torch.norm(h_n, dim=-1, keepdim=True)

			return {ntype: h_n}


class PinSAGE(nn.Module):
	def __init__(self, feature_dim, layer_dim, delta, p):
		super().__init__()

		self.delta = delta
		self.dropout = nn.Dropout(p)

		# Create layers
		in_dim = feature_dim
		self.pinsage_layers = nn.ModuleList()
		for out_dim in layer_dim:
			self.pinsage_layers.append(PinSAGELayer(in_dim, out_dim))
			in_dim = out_dim

		out_dim = layer_dim[-1]
		self.g_1 = nn.Linear(out_dim, out_dim)
		self.g_2 = nn.Linear(out_dim, out_dim, bias=False)
		self.activation = nn.ReLU()

		self.embeddings = None

		# Init
		gain = nn.init.calculate_gain('relu')
		nn.init.xavier_uniform_(self.g_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.g_2.weight, gain=gain)
		nn.init.constant_(self.g_1.bias, 0)

	def forward(self, users, items, bpr, g):
		item_embs = self.embeddings[items]
		if bpr:
			user_embs = self.embeddings[users]

			predictions = torch.matmul(user_embs, item_embs.T)
		else:
			g = g.to(self.embeddings.device)
			with g.local_scope():
				g.nodes['item'].data['h'] = item_embs
				g.pull(users, fn.copy_u('h', 'm'), fn.sum('m', 'h'), etype='iu')
				predictions = torch.matmul(g.nodes['user'].data['h'][users], item_embs.T)

				# v, u, eid = g.in_edges(users, 'all', etype='iu')  # Get items rated by users u.
				# v, reverse = torch.unique(v, return_inverse=True)  # For efficiency, only consider unique items.
				# sims = torch.einsum('vd,id->vi', self.embeddings[v], item_embs)  # Calculate similarities rated items and unrated.
				# g = g.edge_subgraph({'iu': eid}).to(sims.device)  # Subgraph with users and rated items only.
				# g.edges['iu'].data['s'] = sims[reverse]  # Revert to original order and assign as similarity for edges.
				# g.multi_update_all({'iu': (fn.copy_e('s', 'm'), fn.sum('m', 'h'))}, 'sum')  # Represent user based on rated items similarity to items.
				# predictions = g.nodes['user'].data['h']  # The representation is actally a ranking.
				# test = torch.zeros((len(users), len(items))).to(predictions.device)

				# pred = 0
				# m = predictions.mean(dim=0)
				# for i, user in enumerate(users):
				# 	if user in u:
				# 		test[i] = predictions[pred]
				# 		pred += 1
				# 	else:
				# 		test[i] = m

			# predictions = test

		return predictions

	def embedder(self, blocks, ntype='_N'):
		block = blocks[0]
		x = {ntype: self.dropout(block.nodes[ntype].data['feats']) for ntype in block.ntypes}
		# Algorithm 2 line 8 to 14.
		for i, (block, layer) in enumerate(zip(blocks, self.pinsage_layers)):
			x = layer(block, x, ntype=ntype)

		# Algorithm 2 line 16
		x = self._mlp(x, ntype)

		return x

	def _mlp(self, x, ntype=None):
		# Algorithm 2 line 16
		x = self.g_2(self.activation(self.g_1(x[ntype] if ntype is not None else x)))

		return x

	def predict(self, g: dgl.DGLGraph, features):
		with g.local_scope():
			users, items = g.edges('uv')
			p = torch.mul(features[users], features[items]).sum(dim=-1)
			return p

	def loss(self, pos_graph: dgl.DGLGraph, neg_graph: dgl.DGLGraph, blocks: List[dgl.DGLGraph], bpr: bool):
		# Get embeddings
		embeddings = self.embedder(blocks, '_N' if bpr else 'item')

		# Predict pos and neg.
		pos = self.predict(pos_graph, embeddings)
		neg = self.predict(neg_graph, embeddings)

		# Loss equation 1.
		if bpr:
			loss = - torch.nn.functional.logsigmoid(pos - neg).mean()
		else:
			neg = neg.reshape(len(pos), -1)
			pos = pos.unsqueeze(-1)
			loss = torch.clamp(neg - pos + self.delta, min=0).mean()
		return loss, pos > neg

	def inference(self, g, nodes, sampler_args, bpr, use_cuda, batch_size=1024, ntype='_N'):
		device = self.g_1.weight.device
		if bpr:
			sampler = PinSAGESampler(g, 1, *sampler_args, prefetch_node_feats=['feats'])
			dataloader = DataLoader(
				g, nodes, sampler, device=device, batch_size=batch_size,
				shuffle=False, drop_last=False, use_uva=use_cuda
			)
		else:
			sampler = PinSAGESampler2(g, 1, *sampler_args, prefetch_node_feats=['feats'])
			dataloader = DataLoader(
				g, nodes, sampler, device=device, batch_size=batch_size,
				shuffle=False, drop_last=False, use_uva=use_cuda
			)

		embeddings = g.nodes[ntype].data['feats'].to(device)
		for layer in self.pinsage_layers:
			next_embedding = []
			for input_nodes, output, (block, ) in dataloader:
				next_embedding.append(layer(block, {ntype: embeddings[input_nodes]}, ntype=ntype)[ntype])

			embeddings = torch.cat(next_embedding)

		next_embedding = []
		for _, output, _ in dataloader:
			next_embedding.append(self._mlp(embeddings[output]))

		self.embeddings = torch.cat(next_embedding)
