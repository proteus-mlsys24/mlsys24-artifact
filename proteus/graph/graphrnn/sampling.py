from typing import List
import random
import numpy as np
import networkx as nx
import scipy.stats as stats

from .statistics import topology_features, compute_topology_statistics
from .graphrnn_utils import load_and_process_graphrnn_graphs, process_batch, prune_graph_new

class TopologySolutionSet:
    def __init__(self, topologies, lows, highs):
        self.topologies = topologies
        self.lows = lows
        self.highs = highs


class TopologyPicker:
    def __init__(self, graphs: List[nx.DiGraph]):
        self.graphs = graphs

        # remove single node topologies
        checker = lambda graph: len(graph.nodes) > 2
        self.graphs = list(filter(checker, self.graphs))
        print(f"After filtering size we have {len(self.graphs)} graphs left.")

        # ensure satisfiability
        # this is no longer needed since we check for sat during sampling
        # self.graphs = list(filter(topology_is_satisfiable, tqdm(self.graphs, leave=False, desc="sat")))
        # print(f"After filtering satisfiability we have {len(self.graphs)} graphs left.")

        random.shuffle(self.graphs)

        self.graph_uses = {idx: 1 for idx in range(len(self.graphs))}
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

        # prior stats about the empirical distribution of models
        self.prior_stats = {s: [] for s in topology_features}
        self.prior_stat_fns = {}

    def get_graph_feature_vector(self, graph: nx.DiGraph):
        graph_stats = compute_topology_statistics(graph)
        stat_vector = []
        for stype in topology_features:
            stat_vector.append(graph_stats[stype] / self.stds[stype])
        return np.array(stat_vector)

    # This is the function to call if you want similar topologies.
    def sample_adjacent_graphs(self, graph: nx.DiGraph, count: int, beta=0.1):
        print(f"Using features: {topology_features}")
        x = self.get_graph_feature_vector(graph)
        print(f"Provided graph stats: {x}")

        # position of the center
        alphas = np.random.rand(len(topology_features)) * beta
        print(f"alphas: {alphas}, beta: {beta}")

        # derive range from the center
        lows = x - alphas
        highs = lows + beta
        print(f"lows: {lows}, highs: {highs}")
        return self.importance_sampling_uniform(graph, count, lows, highs)

    def importance_sampling_uniform(self, graph: nx.DiGraph, count: int, lows, highs):
        # find the required number of inputs and outputs
        graph_inputs = len(list(filter(lambda node: graph.in_degree(node) == 0, graph.nodes)))
        graph_outputs = len(list(filter(lambda node: graph.out_degree(node) == 0, graph.nodes)))

        # find the available graphs that fall within this limit
        candidates = []
        ps = []
        for idx, graph in enumerate(self.graphs):
            feats = self.get_graph_feature_vector(graph)
            if (feats >= lows).all() and (feats <= highs).all():
                candidates.append(graph)
                uses_factor = 1 / self.graph_uses[idx]
                if self.graph_uses[idx] > 1:
                    print(f"This graph {idx} already has {self.graph_uses[idx]} uses.")

                ps.append(uses_factor / self.joint_prob(feats))

        print(f"Found {len(candidates)} candidates.")

        # normalize probabilities
        ps = np.array(ps).squeeze()
        ps /= np.sum(ps)

        if len(candidates) == 0: return None

        # sample graphs
        tries = 0
        while True:
            indices = np.random.choice(np.arange(len(candidates)), min(len(candidates), count), replace=False, p=ps)
            indices = indices.tolist()
            print(f"Picked indices {indices}")


            topologies = [candidates[idx] for idx in indices]
            tries += 1
            # if tries < 100 and len(topologies) < count: continue

            for idx in indices: self.graph_uses[idx] += 1
            return TopologySolutionSet(topologies, lows, highs)
