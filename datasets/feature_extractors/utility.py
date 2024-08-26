import networkx as nx


def extract_degree(graph: nx.Graph, node):
    return graph.degree(node) if node in graph else 0