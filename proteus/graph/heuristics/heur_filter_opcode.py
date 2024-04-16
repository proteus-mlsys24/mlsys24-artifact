from proteus.graph import Graph, Placeholder, OperatorRegistry
from proteus.config import ProteusConfig

import z3
import pickle
import onnx
from typing import Dict, Set
import numpy as np
import networkx as nx

opcode_frequencies: Dict[str, int]
with open(ProteusConfig.opcode_freqs_path, "rb") as fp:
    opcode_frequencies = pickle.load(fp)

    opcodes = list(filter(lambda opcode: opcode in OperatorRegistry.operator_ids, opcode_frequencies.keys()))
    frequency_total = sum(map(lambda opcode: opcode_frequencies[opcode], opcodes))
    opcode_probs = [opcode_frequencies[opcode] / frequency_total for opcode in opcodes]


def apply_heuristic(G: Graph, whitelist: Set[str] = None):
    np.random.seed()

    if whitelist:
        selected_opcodes = list(whitelist)
    else:
        n_opcodes = 12
        selected_indices = np.random.choice(np.arange(len(opcodes)), size=n_opcodes, replace=False, p=opcode_probs)
        selected_opcodes = [opcodes[i] for i in selected_indices]
        print("selected_opcodes", selected_opcodes)

    # whitelist_ids = [OperatorRegistry.operator_ids[opcode] for opcode in selected_opcodes]

    for node in G.nodes:
        node.whitelist = set(selected_opcodes)

    # cons = []
    # for node in G.nodes:
    #     if isinstance(node, Placeholder):
    #         cons.append(z3.Or([node.node_opcode == opcode_id for opcode_id in whitelist_ids]))
    # return cons


def collect_opcodes(model: onnx.ModelProto):
    opcodes = set()
    for node in model.graph.node:
        opcodes.add(node.op_type)

    return opcodes

def collect_opcodes_networkx(model: nx.DiGraph):
    model_opcodes = set()
    for node in model.nodes:
        opcode = model.nodes[node]["opcode"]
        model_opcodes.add(opcode)

    return model_opcodes