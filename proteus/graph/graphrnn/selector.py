import random
import pickle
import os
import networkx as nx
from typing import List, Dict, Tuple, Optional

from proteus.graph import Graph, Placeholder, Edge
from proteus.config import ProteusConfig
from .graphrnn_utils import load_and_process_graphrnn_graphs, process_batch, prune_graph_new
from .sampling_new import TopologyDistribution
from .sampling import TopologyPicker, TopologySolutionSet


def canonicalize_shape(shape: List[int]):
    padded = list(shape)
    while len(padded) < Placeholder.max_dims: padded.insert(0, 0)
    return padded


class SimpleTopologySelector:
    def __init__(self, graph_paths: Optional[List[str]] = None):
        self.graphs: List[nx.DiGraph] = list()

        if graph_paths is None:
            for graph_path in ProteusConfig.graphrnn_graphs:
                graphs = load_and_process_graphrnn_graphs(graph_path, is_real=False)
                self.graphs.extend(graphs)
        else:
            for graph_path in graph_paths:
                graphs = load_and_process_graphrnn_graphs(graph_path, is_real=False)
                self.graphs.extend(graphs)


    def select_topology(self, original: nx.DiGraph, num_inputs: int, num_outputs: int):
        while True:
            graph: nx.DiGraph = random.choice(self.graphs)

            graph_inputs = len(list(filter(lambda node: graph.in_degree(node) == 0, graph.nodes)))  # noqa
            graph_outputs = len(list(filter(lambda node: graph.out_degree(node) == 0, graph.nodes)))  # noqa

            if graph_inputs >= num_inputs and graph_outputs >= num_outputs:
                return prune_graph_new(graph, num_inputs, num_outputs)

    def convert_topology(self, original: Graph):
        # select a topology like the original (has the same inputs and outputs)
        input_nodes = list(original.graph_input_nodes())
        output_nodes = list(original.graph_output_nodes())

        input_node_shapes = [
            node.input_shapes
            for node in original.graph_input_nodes()
        ]

        output_node_shapes = [
            node.output_shape
            for node in original.graph_output_nodes()
        ]

        topology = self.select_topology(original, len(input_nodes), len(output_nodes))

        return self.construct_topology(topology, input_node_shapes, output_node_shapes)


    def convert_topology_networkx(self, original: nx.DiGraph):
        input_nodes = list(filter(lambda node: original.in_degree(node) == 0, original.nodes))
        output_nodes = list(filter(lambda node: original.out_degree(node) == 0, original.nodes))

        input_node_shapes = [
            original.nodes[node]["input_shapes"]
            for node in input_nodes
        ]

        output_node_shape = [
            original.nodes[node]["output_shape"]
            for node in output_nodes
        ]

        topology = self.select_topology(original, len(input_nodes), len(output_nodes))

        return self.construct_topology(topology, input_node_shapes, output_node_shape)



    def construct_topology(self, topology: nx.DiGraph, input_node_shapes: List[List[Tuple[int]]], output_node_shapes: list[Tuple[int]]):
        # construct a new graph with placeholders that has matching input and output
        # dimensions as the original topology
        G = Graph()
        constraints = []
        converted_nodes: Dict[int, Placeholder] = dict()

        # handle the input and output nodes first. also collect shape constraints
        for idx, input_node in enumerate(filter(lambda node: topology.in_degree(node) == 0, topology.nodes)):
            node = Placeholder(len(input_node_shapes[idx]), topology.out_degree(input_node))
            G.nodes.append(node)
            converted_nodes[input_node] = node

            # make constraints
            for input_id, input_shape in enumerate(input_node_shapes[idx]):
                padded_shape = canonicalize_shape(input_shape)
                for dim_id in range(Placeholder.max_dims):
                    constraints.append(node.input_dims[input_id][dim_id] == padded_shape[dim_id])

        for idx, output_node in enumerate(filter(lambda node: topology.out_degree(node) == 0, topology.nodes)):
            node = Placeholder(topology.in_degree(output_node), 1)
            G.nodes.append(node)
            converted_nodes[output_node] = node

            # make constraints
            desired_output_shape = canonicalize_shape(output_node_shapes[idx])
            for dim_id in range(Placeholder.max_dims):
                constraints.append(node.output_dims[0][dim_id] == desired_output_shape[dim_id])

        # then add the remaining nodes
        for node in topology.nodes:
            if node in converted_nodes: continue
            converted = Placeholder(topology.in_degree(node), topology.out_degree(node))
            G.nodes.append(converted)
            converted_nodes[node] = converted

        # add the edges into the graph
        in_count = {node: 0 for node in topology.nodes}
        out_count = {node: 0 for node in topology.nodes}
        for u, v in topology.edges:
            src_idx, dst_idx = out_count[u], in_count[v]
            out_count[u] += 1
            in_count[v] += 1

            G.edges.append(Edge(converted_nodes[u], converted_nodes[v], src_idx, dst_idx))

        return topology, G, constraints


class ImportanceSamplingTopologySelector(SimpleTopologySelector):
    # def __init__(self):
    def __init__(self, graph_paths: Optional[List[str]] = None):
        super().__init__(graph_paths)
        self.samplers: Dict[Tuple[int, int], TopologyDistribution] = dict()

    def create_or_load_distribution(self, num_inputs: int, num_outputs: int):
        os.makedirs("distrib_cache", exist_ok=True)
        filename = f"distrib_cache/distrib_{num_inputs}_{num_outputs}.pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as fp:
                distrib = pickle.load(fp)
                return distrib

        distrib = TopologyDistribution(self.graphs, num_inputs, num_outputs, 5)
        with open(filename, "wb") as fp:
            pickle.dump(distrib, fp)

        return distrib

    def select_topology(self, original: nx.DiGraph, num_inputs: int, num_outputs: int):
        if (num_inputs, num_outputs) not in self.samplers:
            self.samplers[(num_inputs, num_outputs)] = self.create_or_load_distribution(num_inputs, num_outputs)

        distrib = self.samplers[(num_inputs, num_outputs)]
        # candidates = distrib.sample_similar(original.make_networkx())
        for beta in [1, 2, 4, 8]:
            candidates = distrib.sample_similar(original, beta=beta)
            if candidates:
                return random.choice(candidates)

        raise Exception("Failed to find any topologies.")