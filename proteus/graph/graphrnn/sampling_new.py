from typing import List
import networkx as nx
import numpy as np
import scipy.stats as stats
import functools
import random
from tqdm import tqdm

from proteus.graph.graphrnn.graphrnn_utils import prune_graph_new, ensure_num_inputs_outputs_connected
from .statistics import topology_features, compute_topology_statistics


class TopologyDistribution:
    def __init__(self, graphs: List[nx.DiGraph], num_inputs: int, num_outputs: int, num_samples: int, tqdm_position=None):
        print(f"TopologyDistribution. num_inputs: {num_inputs}, num_outputs: {num_outputs}")
        assert num_inputs > 0 and num_outputs > 0, "Invalid number of inputs or outputs."

        # populate the graphs
        self.graphs = []
        if tqdm_position is not None:
            graphs = tqdm(graphs, position=tqdm_position)
        for graph in graphs:
            if random.random() < 0.5:
                graph = graph.reverse()

            # graph_inputs = len(list(filter(lambda node: graph.in_degree(node) == 0, graph.nodes)))  # noqa
            # graph_outputs = len(list(filter(lambda node: graph.out_degree(node) == 0, graph.nodes)))  # noqa
            #
            # if graph_inputs < num_inputs or graph_outputs < num_outputs: continue

            for rep in range(num_samples):
                # self.graphs.append(prune_graph_new(graph, num_inputs, num_outputs))
                pruned = ensure_num_inputs_outputs_connected(graph, num_inputs, num_outputs)
                if pruned and nx.is_weakly_connected(pruned):
                    self.graphs.append(pruned)


        print(f"Initial candidates: {len(graphs)}. Filtered candidates: {len(self.graphs)}")

        # compute graph statistics
        self.graph_stats = {s: [] for s in topology_features}

        for idx, graph in enumerate(self.graphs):
            graph_stats = compute_topology_statistics(graph)
            for stype in topology_features:
                self.graph_stats[stype].append(graph_stats[stype])

        self.stds = {s: np.std(self.graph_stats[s]) for s in topology_features}
        self.normalized_stats = {
            k: np.array(v) / self.stds[k]
            for k, v in self.graph_stats.items()
        }

        # The joint probability distribution, used for importance sampling
        dataset = np.array(list(map(self.get_graph_feature_vector, self.graphs))).T
        self.joint_prob = stats.gaussian_kde(dataset)

    @functools.cache
    def get_graph_feature_vector(self, graph: nx.DiGraph):
        graph_stats = compute_topology_statistics(graph)
        stat_vector = []
        for stype in topology_features:
            stat_vector.append(graph_stats[stype] / self.stds[stype])
        return np.array(stat_vector)

    def importance_sampling(self, graphs: List[nx.DiGraph], count: int):
        weights: List[float] = []

        for graph in graphs:
            features = self.get_graph_feature_vector(graph)
            prob = self.joint_prob(features)
            weights.append(1 / prob)

        normalized_weights = np.array(weights).flatten()
        normalized_weights /= normalized_weights.sum()

        indices = np.random.choice(range(len(graphs)), min(count, len(graphs)), replace=False, p=normalized_weights)

        return [graphs[idx] for idx in indices]

    def find_adjacent(self, graph: nx.DiGraph, beta: float):
        x = self.get_graph_feature_vector(graph)

        alphas = np.random.rand(len(topology_features)) * beta

        # derive range from the center
        lows = x - alphas
        highs = lows + beta

        def filter_fn(graph: nx.DiGraph):
            feats = self.get_graph_feature_vector(graph)
            return (feats >= lows).all() and (feats <= highs).all()

        return list(filter(filter_fn, self.graphs))

    def sample_similar(self, graph: nx.DiGraph, beta=2, count=10):
        candidates = self.find_adjacent(graph, beta)
        if len(candidates) > 0:
            return self.importance_sampling(candidates, count)
