from functools import lru_cache

from torch import nn
import scipy
import networkx as nx


class PersonalizedPagerank(nn.Module):
    def __init__(self, graph, alpha):
        super().__init__()
        self.graph = graph
        self.alpha = alpha
        self.max_iteration = 100
        self.tolerance = 1.0e-6
        self.node_list = graph.nodes()
        self.node_set = set(self.node_list)

        self.N = len(graph)
        self.M = nx.to_scipy_sparse_matrix(graph, nodelist=self.node_list, weight='weight', dtype=float)

        S = scipy.array(self.M.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = scipy.sparse.spdiags(S.T, 0, *self.M.shape, format='csr')

        self.M = Q * self.M

    @lru_cache(maxsize=4048)
    def forward(self, source_nodes):
        source_nodes = list(source_nodes)
        personalization = {entity: 1 for entity in source_nodes} if source_nodes else None

        # Initialize with equal PageRank to each node
        x = scipy.repeat(1.0 / self.N, self.N)

        # Personalization vector
        if not personalization:
            p = scipy.repeat(1.0 / self.N, self.N)
        else:
            p = scipy.array([personalization.get(n, 0) for n in self.node_list], dtype=float)
            p = p / p.sum()

        # power iteration: make up to max_iter iterations
        for _ in range(self.max_iteration):
            last_x = x
            x = self.alpha * (x * self.M) + (1 - self.alpha) * p
            # check convergence, l1 norm
            err = scipy.absolute(x - last_x).sum()
            if err < self.N * self.tolerance:
                return dict(zip(self.node_list, map(float, x)))
        raise RuntimeError('PageRank failed to converge')