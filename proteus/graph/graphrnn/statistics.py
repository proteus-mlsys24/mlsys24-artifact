import networkx as nx
import numpy as np
from networkx.algorithms.approximation.clustering_coefficient import average_clustering


def average_degree(graph):
    degrees = list(d for n, d in graph.degree())
    return sum(degrees) / len(degrees)


def clustering_coefficient(graph):
    # return average_clustering(graph.to_undirected())
    return np.average(list(nx.clustering(graph.to_undirected()).values()))


def graph_diameter(graph):
    return nx.diameter(graph.to_undirected())


def num_nodes(graph):
    return len(graph.nodes)


topology_statistic_fns = {
    "average_degree": average_degree,
    # "clustering_coefficient": clustering_coefficient,
    "diameter": graph_diameter,
    "num_nodes": num_nodes
}

topology_features = list(sorted(topology_statistic_fns.keys()))


def compute_topology_statistics(graph: nx.DiGraph):
    return {
        s: topology_statistic_fns[s](graph)
        for s in topology_features
    }
