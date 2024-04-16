from proteus.graph import Graph, Placeholder, FixedNode, GraphNode, OperatorRegistry
from proteus.graph.onnx.utils import extract_attributes
from queue import Queue
from typing import List, Tuple, Dict, Callable, Optional
import z3


def get_paths(G: Graph, pathlen: int, all_paths=False):
    q = Queue()
    paths = []

    # collect next nodes
    next_nodes = dict()
    for node in G.nodes:
        next_nodes[node] = list()
    for edge in G.edges:
        next_nodes[edge.src].append(edge.dst)

    # add initial nodes into queue
    for node in G.nodes: q.put([node, ])

    # traverse the graphs to gather paths
    while not q.empty():
        cur_path = q.get()

        if len(cur_path) == pathlen:
            if all_paths or any((isinstance(node, Placeholder) for node in cur_path)):
                paths.append(cur_path)
            continue

        for next_node in next_nodes[cur_path[-1]]:
            new_path = cur_path[:]
            new_path.append(next_node)
            q.put(new_path)

    return paths


def node_to_str(node):
    opcode = node.onnx_node.op_type
    attrs = extract_attributes(node.onnx_node)
    if "kernel_shape" in attrs:
        return f"{opcode}_{max(attrs['kernel_shape'])}"
    return opcode

def placeholder_to_str(node, model):
    opcode_id = model[node.node_opcode].as_long()
    opcode_name = OperatorRegistry.operators[opcode_id]
    
    if opcode_name == "Conv":
        candidate = node.opcode_candidates[opcode_name]
        kernel_size = model[candidate.kernel_size].as_long()
        
        return f"{opcode_name}_{kernel_size}"
    
    return opcode_name

def opcode_sequences(G: Graph, pathlen: int):
    paths = get_paths(G, pathlen, all_paths=True)
    opcode_seqs = []
    for path in paths:
        if any((isinstance(node, Placeholder) for node in path)): continue
        opcode_seqs.append(list(map(node_to_str, path)))

    return opcode_seqs


def opcode_frequency(G: Graph):
    freqs = dict()
    for node in G.nodes:
        if isinstance(node, Placeholder): continue
        # opcode = node.onnx_node.op_type
        opcode = node_to_str(node)
        if opcode not in freqs: freqs[opcode] = 0
        freqs[opcode] += 1

    return freqs


def list_to_frequency(lst):
    freqs = dict()
    for elem in lst:
        if isinstance(elem, list): elem = tuple(elem)
        if elem not in freqs: freqs[elem] = 0
        freqs[elem] += 1

    return freqs


def merge_frequencies(f1, f2):
    freqs = dict()
    for k in f1:
        if k not in freqs: freqs[k] = 0
        freqs[k] += f1[k]

    for k in f2:
        if k not in freqs: freqs[k] = 0
        freqs[k] += f2[k]

    return freqs


def path_conditional_probability(path: Tuple[GraphNode], model: z3.ModelRef, frequencies: Dict[Tuple[str], int],
                                 process_fn: Optional[Callable[[List[str]], List[str]]] = None):
    placeholder_mask = [isinstance(node, Placeholder) for node in path]
    completed_path: List[str] = []
    for node in path:
        if isinstance(node, FixedNode):
            completed_path.append(node.onnx_node.op_type)
        else:
            # opcode_id = model[node.node_opcode].as_long()
            # opcode_name = OperatorRegistry.operators[opcode_id]
            # completed_path.append(opcode_name)
            label = placeholder_to_str(node, model)
            completed_path.append(label)

    if process_fn is not None:
        completed_path = process_fn(completed_path)

    numerator, denominator = 1, 1

    for ref, count in frequencies.items():
        match_exact = all(r == p for (r, p) in zip(ref, completed_path))
        match_masked = all((m or r == p for (m, r, p) in zip(placeholder_mask, ref, completed_path)))

        if match_exact: numerator += count
        if match_masked: denominator += count

    return numerator, denominator
