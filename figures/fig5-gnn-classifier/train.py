
import torch
import itertools
import pickle
import random
from typing import List, Set
from dataset import ModelDataset
from gcn import GCN
import pytorch_lightning as pl
import networkx as nx
from torch_geometric.data import DataLoader

def filter_fn(graph: nx.DiGraph):
    return len(graph.nodes) > 1

def load_consolidated_fakes(filename):
    with open(filename, "rb") as fp:
        graphs = pickle.load(fp)
        graphs = list(filter(filter_fn, graphs))
    
    labels = [ModelDataset.LABEL_FAKE] * len(graphs)
    return graphs, labels

def load_reals(filename):
    with open(filename, "rb") as fp:
        reals = pickle.load(fp)
        reals = list(itertools.chain(*reals))
        graphs = list(map(lambda n: n[0], reals))
        graphs = list(filter(filter_fn, graphs))
    

    labels = [ModelDataset.LABEL_REAL] * len(graphs)
    return graphs, labels
    
def chain(lst):
    graphs = []
    labels = []
    for g, l in lst:
        graphs.extend(g)
        labels.extend(l)

    return graphs, labels

def collect_opcodes(graphs: List[nx.DiGraph]):
    opcodes: Set[str] = set()
    for graph in graphs:
        for node in graph.nodes:
            opcode = graph.nodes[node]["opcode"]
            if opcode not in opcodes:
                opcodes.add(opcode)
    return opcodes

def scramble_opcodes(graphs: List[nx.DiGraph], labels: List[int], opcodes: Set[str]):
    for g, l in zip(graphs, labels):
        if l == ModelDataset.LABEL_REAL: continue
        for node in g.nodes:
            g.nodes[node]["opcode"] = random.choice(opcodes)
