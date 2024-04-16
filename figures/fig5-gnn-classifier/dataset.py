import torch
import torch.nn.functional as F
import z3
import os
import random
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Union, Tuple, Optional
from functools import partial
from sklearn.utils import shuffle

from torch_geometric.data import Data, Dataset
from proteus.graph import Graph, GraphNode, Placeholder, FixedNode, OperatorRegistry
from proteus.graph.solver import ModelLike


def graph_node_features(node: GraphNode, model: ModelLike):
    if isinstance(node, Placeholder):
        assert model, "model cannot be None when the graph contains placeholders"
        node_opcode = model[node.node_opcode].as_long()
    else:
        node_opcode = OperatorRegistry.operator_ids[node.onnx_node.op_type]

    return F.one_hot(torch.LongTensor([node_opcode]), num_classes=len(OperatorRegistry.operators))


def graph_to_tg_data(G: Graph, model: ModelLike):
    # construct node features
    node_features = []
    node_indices: Dict[GraphNode, int] = dict()
    for idx, node in enumerate(G.nodes):
        node_features.append(graph_node_features(node, model))
        node_indices[node] = idx

    # construct edges
    edges = []
    for edge in G.edges:
        srcidx = node_indices[edge.src]
        dstidx = node_indices[edge.dst]
        edges.append([srcidx, dstidx])

    return Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edges)
    )

def digraph_to_tg_data(G: nx.DiGraph, opcode_lookup: Dict[str, int]):
    G = nx.convert_node_labels_to_integers(G)
    node_opcodes = [
        opcode_lookup[G.nodes[node]["opcode"]] 
        for node in G.nodes
    ]

    node_features = F.one_hot(torch.LongTensor(node_opcodes), num_classes=len(opcode_lookup)).float()
    edges = np.array([[u,v] for u,v in G.edges]).T
    edge_index = torch.from_numpy(edges).long()
    if len(edge_index.shape) == 1: edge_index = edge_index.unsqueeze(0).reshape((2, 0))

    return Data(
        x=node_features,
        edge_index=edge_index
    )

class ModelDataset(Dataset):

    LABEL_REAL = 0
    LABEL_FAKE = 1

    def __init__(self, graphs: List[nx.DiGraph], labels: List[int], filter_fn=None, opcode_indices=None):
        super().__init__()
        self.graphs: List[nx.DiGraph] = graphs
        self.labels: List[int] = labels

        # gather opcodes
        if opcode_indices:
            self.opcode_indices = opcode_indices
        else:
            self.opcodes: Set[str] = set()
            for graph in self.graphs:
                for node in graph.nodes:
                    opcode = graph.nodes[node]["opcode"]
                    if opcode not in self.opcodes:
                        self.opcodes.add(opcode)

            self.opcode_indices: Dict[str, int] = {
                opcode: idx for idx, opcode in enumerate(sorted(self.opcodes))
            }

        # apply filtering
        if filter_fn:
            accept = list(map(filter_fn, self.graphs))
            self.graphs = [ g for (g,a) in zip(graphs, accept) if a ]
            self.labels = [ l for (l,a) in zip(labels, accept) if a ]

        self.graphs, self.labels = shuffle(self.graphs, self.labels)

        # tensorize node features
        self.data = list(map(partial(digraph_to_tg_data, opcode_lookup=self.opcode_indices), self.graphs))

    def len(self):
        return len(self.graphs)

    def get(self, index: int):
        return self.data[index], torch.Tensor([self.labels[index]]).float()

