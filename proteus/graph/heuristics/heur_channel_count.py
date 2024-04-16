import onnx
import z3
from typing import Optional, List
import networkx as nx

from proteus.graph import Graph, FixedNode, OperatorRegistry
import proteus.graph.ops as ops

# COMMON_CHANNEL_COUNTS = [3, 24, 48, 192, 224, ] + [int(2 ** i) for i in range(12)]
COMMON_CHANNEL_COUNTS = []
CONV_ID = OperatorRegistry.operator_ids["Conv"]
CONV_TRANSPOSE_ID = OperatorRegistry.operator_ids["ConvTranspose"]


def apply_heuristic(G: Graph, extra_channels: Optional[List[int]]):
    cons = []
    channels = set(COMMON_CHANNEL_COUNTS)
    if extra_channels:
        channels.update(extra_channels)

    for node in G.nodes:
        if isinstance(node, FixedNode): continue

        if CONV_ID in node.opcode_candidates:
            pl: ops.Conv = node.opcode_candidates[CONV_ID]
            cons.append(z3.Or(*[pl.in_channels == c for c in channels]))
            cons.append(z3.Or(*[pl.out_channels == c for c in channels]))

        if CONV_TRANSPOSE_ID in node.opcode_candidates:
            pl2: ops.ConvTranspose = node.opcode_candidates[CONV_TRANSPOSE_ID]
            cons.append(z3.Or(*[pl2.in_channels == c for c in channels]))
            cons.append(z3.Or(*[pl2.out_channels == c for c in channels]))

    return cons


def get_onnx_channel_counts(model: onnx.ModelProto):
    G = Graph.from_onnx(model)
    channel_counts = []

    for node in G.nodes:
        if node.onnx_node.op_type not in ["Conv", "ConvTranspose"]: continue
        in_channels = node.input_shapes[1][0]
        out_channels = node.input_shapes[1][1]

        channel_counts.append(in_channels)
        channel_counts.append(out_channels)

    return set(channel_counts)


def get_networkx_channel_counts(graph: nx.DiGraph):
    channel_counts = set()
    for node in graph.nodes:
        node_attrs = graph.nodes[node]

        if "Conv" in node_attrs["opcode"]:
            in_channels = node_attrs["input_shapes"][1][0]
            out_channels = node_attrs["input_shapes"][1][1]
            channel_counts.add(in_channels)
            channel_counts.add(out_channels)

    return channel_counts
