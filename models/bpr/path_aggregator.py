import torch
from torch import nn


class RelationAggregator(nn.Module):
	def __init__(self, rel_dim, out_dim, attention_heads, path_length, n_anchors, n_relations):
		super().__init__()
		self.out_dim = out_dim
		self.attention_heads = attention_heads
		self.path_length = path_length
		self.n_anchors = n_anchors
		self.w_p = nn.Linear(rel_dim*path_length, out_dim)  # Path embedding

		self.w_ns = nn.ModuleList()
		self.atts = nn.ModuleList()
		for _ in range(attention_heads):
			self.w_ns.append(nn.Linear(out_dim, out_dim))
			self.atts.append(nn.Linear(out_dim, 1))

		self.r_emb = nn.Embedding(n_relations, rel_dim, padding_idx=0)
		self.w_o = nn.Linear(out_dim*attention_heads, out_dim)
		self.activation = nn.LeakyReLU()

		self.reset()

	def reset(self):
		for param in self.parameters():
			if isinstance(param, nn.Linear):
				nn.init.xavier_uniform_(param.weight)
				nn.init.zeros_(param.bias)
			elif isinstance(param, nn.Embedding):
				nn.init.xavier_uniform_(param.weight)
				param.weight.data[0] = -1e10
				param.weight.data[1] = 1e10
			elif isinstance(param, nn.ModuleList):
				for p in param:
					nn.init.xavier_uniform_(p.weight)
					nn.init.zeros_(p.bias)

	def forward(self, x):
		bs = x.size(0)  # Get batch size

		# Flatten
		x1 = x.reshape(-1)

		# Get relation features
		x2 = self.r_emb(x1)

		# Transform to B x A x P*R  (batch size, no. anchor, path length, relation dim, respectively)
		x3 = x2.reshape(bs * self.n_anchors, -1)

		x = self.activation(self.w_p(x3))
		atts = []
		for w_n, att in zip(self.w_ns, self.atts):
			x_a = self.activation(w_n(x))
			x_a = att(x_a).reshape(bs, -1)
			x_a = torch.softmax(x_a, dim=-1)
			x_a = x * x_a.reshape(-1, 1)
			atts.append(x_a.reshape(bs, self.n_anchors, -1).sum(1))

		x = torch.cat(atts, dim=1)
		x = self.w_o(x)
		norm = torch.norm(x, p=1, dim=1)

		return x / norm.unsqueeze(-1)
		