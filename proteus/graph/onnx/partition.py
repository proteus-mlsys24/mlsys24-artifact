import copy
import onnx
import random
import math
import graphviz
from os import getpid
from typing import List
from time import time, perf_counter_ns

import numpy as np
from multiprocessing import Pool

def printc(text):
    from termcolor import colored
    print(colored(text, "yellow"))

class Partition:
    def __init__(self, model, subgraph, inputs, outputs):
        self.model: onnx.ModelProto = model
        self.subgraph: onnx.GraphProto = subgraph
        self.inputs = inputs
        self.outputs = outputs


class PartitionSet:
    def __init__(self, model, partitions):
        self.model = model
        self.partitions: List[Partition] = partitions
        self.edges = set()

    # get the partition sizes
    def partition_sizes(self):
        return [ len(p.subgraph.node) for p in self.partitions ]
    
    # get the std of partition sizes
    def partition_sizes_std(self):
        return np.std(self.partition_sizes())

    # connect the output (src, srcIdx) to the input (dst, dstIdx)
    def connect(self, src, dst, srcIdx, dstIdx):
        self.edges.add((src, dst, srcIdx, dstIdx))

    # mutate subgraph
    def mutate(self, k, new_model):
        new_subgraph = new_model.graph
        input_names = list(map(lambda node: node.name, new_subgraph.input))
        output_names = list(map(lambda node: node.name, new_subgraph.output))

        print(k,
              "inputs:", list(map(lambda node: node.name, self.partitions[k].subgraph.input)),
              "->", input_names,
              "outputs:", list(map(lambda node: node.name, self.partitions[k].subgraph.input)),
              "->", output_names)

        new_partition = Partition(new_model, new_subgraph, input_names, output_names)
        self.partitions[k] = new_partition

    # reassemble the partitions into a large graph
    def reassemble(self):
        # collect subgraphs and add prefixes
        subgraphs = []
        name_map = dict()
        for i, partition in enumerate(self.partitions):
            subgraph = partition.subgraph
            # subgraph = onnx.compose.add_prefix_graph(subgraph, f"p{i}_", name_map=name_map)

            for o in subgraph.output:
                name_map[o.name] = f"p{i}_{o.name}"
            for o in subgraph.input:
                name_map[o.name] = f"p{i}_{o.name}"

            subgraph = onnx.compose.add_prefix_graph(subgraph, f"p{i}_")
            subgraphs.append(subgraph)

        # create mapping from (output tensor) -> (input tensor)
        # we will later replace all instances of input tensor with their corresponding
        # output tensors.
        input_mapping = dict()
        for (src, dst, srcIdx, dstIdx) in self.edges:
            src_tensor_name = subgraphs[src].output[srcIdx].name
            dst_tensor_name = subgraphs[dst].input[dstIdx].name
            input_mapping[dst_tensor_name] = src_tensor_name

            print(src_tensor_name, "->", dst_tensor_name)

        # find the partition containing the output
        output_partitions = set(range(len(self.partitions)))
        for (src, dst, srcIdx, dstIdx) in self.edges:
            if src in output_partitions:
                output_partitions.remove(src)

        # printc("-"*50)
        # printc(f"output_partitions: {output_partitions}")
        # for idx, part in enumerate(self.partitions):
        #     output_names = list(map(lambda o: o.name, part.model.graph.output))
        #     printc(f"{idx}, {output_names}")
        # printc("-"*50)

        # create a merged graph with all the original vertices
        assembled = onnx.GraphProto()
        assembled.name = "assembled_graph"
        for idx, subgraph in enumerate(subgraphs):
            assembled.node.extend(subgraph.node)
            assembled.initializer.extend(subgraph.initializer)
            assembled.value_info.extend(subgraph.value_info)

        # extract the model outputs from partitions
        original_output_names = [ name_map[output.name] for output in self.model.graph.output ]
        for p in subgraphs:
            for o in p.output:
                if o.name in original_output_names:
                    assembled.output.append(o)

        # for o in output_partitions:
        #     assembled.output.extend(subgraphs[o].output)

        # replace the names
        for node_idx, node in enumerate(assembled.node):
            for input_idx, input_tensor in enumerate(node.input):
                if input_tensor in input_mapping:
                    assembled.node[node_idx].input[input_idx] = input_mapping[input_tensor]

        # find the actual input of the graph
        original_input_names = [ t.name for t in self.model.graph.input ]
        input_index_map = { name_map[t]: idx for idx,t in enumerate(original_input_names) }
        graph_inputs = [ None for i in range(len(original_input_names))]
        for subgraph in subgraphs:
            for input_tensor in subgraph.input:
                if input_tensor.name not in input_mapping:
                    # assembled.input.append(input_tensor)
                    graph_inputs[input_index_map[input_tensor.name]] = input_tensor

        assembled.input.extend(graph_inputs)

        model = onnx.helper.make_model(assembled)

        # don't you love magic numbers?
        model.ir_version = 6
        model.opset_import[0].version = 11
        return model


"""
Randomly partitions the graph with random contraction (i.e. Karger's Algorithm)

model: the graph to partition
k: the number of partitions we want to end up with

returns array of onnx models
"""


def random_contraction(graph: onnx.onnx_ml_pb2.GraphProto, k, min_partition_size=None, approx_equal_sizes=False):
    par = dict()
    psize = dict()

    edges = set()
    outputs = dict()
    for node in graph.node:
        par[node.name] = node.name
        psize[node.name] = 1
        for o in node.output: outputs[o] = node

    for u in graph.node:
        for v in graph.node:
            # (u, v) if an output of u is an input of v
            if len(set(u.output).intersection(set(v.input))) > 0:
                edges.add((u.name, v.name))

    opcode_lookup = dict()
    for node in graph.node:
        opcode = node.op_type
        opcode_lookup[node.name] = opcode

    edges = list(edges)

    # print(f"Collected {len(par)} nodes and {len(edges)} edges.")

    def find_root(node):
        while par[node] != node: node = par[node]
        return node

    def weight_fn(e):
        pa, pb = find_root(e[0]), find_root(e[1])
        if pa == pb: return 0
        
        # prevent partitions from getting too big
        max_size = 2 * int(len(graph.node) / k)
        if psize[pa] >= max_size and psize[pb] >= max_size: return 1e-5
        # if psize[pa] + psize[pb] >= max_size: return 0

        # return 1 / math.exp(psize[pa] + psize[pb])
        # return 1 / max(psize[pa], psize[pb])
        # return math.sqrt(psize[pa] + psize[pb])
        # return 1 / (psize[pa] + psize[pb]) ** 2
        # return 1 / max(psize[pa], psize[pb])
        # return 1 / math.sqrt(psize[pa] + psize[pb])
        return 1 / (psize[pa] + psize[pb])

    def compute_edge_weights(edges):
        component_freqs = dict()
        for (u, v) in edges:
            pa, pb = find_root(u), find_root(v)
            if pa > pb: pa, pb = pb, pa
            if (pa, pb) not in component_freqs:
                component_freqs[(pa, pb)] = 0
            component_freqs[(pa, pb)] += 1

        weights = []
        for e in edges:
            pa, pb = find_root(e[0]), find_root(e[1])
            if pa > pb: pa, pb = pb, pa
            weights.append(weight_fn(e) / component_freqs[(pa, pb)])

        return np.array(weights)


    def sample_edge():
        while True:
            if approx_equal_sizes:
                # each edge has weight 1/sqrt(e[0].size + e[1].size) if e[0] != e[1]
                # or 0 otherwise
                # weights = np.array(list(map(weight_fn, edges)))
                weights = compute_edge_weights(edges)
                weights = weights / weights.sum()

                e_idx = np.random.choice(np.arange(len(edges)), size=1, p=weights)[0]
                e = edges[e_idx]
            else:
                e = random.sample(edges, 1)[0]

            pa, pb = find_root(e[0]), find_root(e[1])
            if pa != pb: return e

    num_partitions = len(par)

    # merge the LayerNorms
    def get_prefix(p):
        return "/".join(p.split("/")[:-1])

    adjacent = set()
    for u, v in edges:
        pu = get_prefix(u)
        pv = get_prefix(v)
        if opcode_lookup[u] == "Add" and pv.endswith("LayerNorm"): 
            adjacent.add(u)

    for u, v in edges:
        pu = get_prefix(u)
        pv = get_prefix(v)

        merge = False

        if pu == pv and pu.endswith("LayerNorm"): merge = True
        if opcode_lookup[u] == "Add" and pv.endswith("LayerNorm"): 
            # print(f"Add Merge: {u} -> {v}.")
            merge = True
        if opcode_lookup[u] == "Add" and v in adjacent:
            # print(f"Add Merge CASE 2: {u} -> {v}.")
            merge = True

        if merge:
            # print("Merging", pu, u, v)
            pa, pb = find_root(u), find_root(v)
            if pa != pb:
                par[pa] = pb
                # psize[pb] += psize[pa]

                num_partitions -= 1

    while num_partitions > k:
        e = sample_edge()
        pa, pb = find_root(e[0]), find_root(e[1])
        if pa != pb:
            par[pa] = pb
            psize[pb] += psize[pa]

        num_partitions -= 1

    # TODO make sure psize calculation is correct
    # print("group sizes", [ psize[i.name] for i in graph.node if par[i.name] == i.name ])

    partitions = dict()
    for node in graph.node:
        root = find_root(node.name)
        if root not in partitions: partitions[root] = list()
        partitions[root].append(node)

    # print(f"We got {len(partitions)} partitions.")
    return partitions


"""
Creates a graphviz dot graph for the partitions

model: the onnx model object
partitions: a mapping from (partition_name) -> List[nodes]
"""
def visualize_partitions(model, partitions):
    dot = graphviz.Digraph()
    for idx, p in enumerate(partitions):
        nodes = partitions[p]
        with dot.subgraph(name=f"cluster_{idx}", graph_attr={"label": f"subgraph {idx}"}) as s:
            for n in nodes:
                s.node(n.name, shape="box")

    for u in model.graph.node:
        for v in model.graph.node:
            # (u, v) if an output of u is an input of v
            if len(set(u.output).intersection(set(v.input))) > 0:
                dot.edge(u.name, v.name)

    return dot


"""
Creates a list of onnx objects based on partition. This can then be
fed into onnxruntime for graph level optimization.

ref: https://github.com/onnx/onnx/issues/2078

model: the onnx model object
partitions: a mapping from (partition_name) -> List[nodes]

returns: PartitionSet
"""


def export_partitions_onnx(model, partitions):
    input_names = set(map(lambda x: x.name, model.graph.input))
    initializer_names = set(map(lambda x: x.name, model.graph.initializer))

    model = onnx.shape_inference.infer_shapes(model)
    model_inputs = set(map(lambda i: i.name, model.graph.input))
    model_outputs = set(map(lambda i: i.name, model.graph.output))

    inputs_of = dict()
    outputs_of = dict()

    onnx_partitions = list()

    # We first find the inputs/outputs of each subgraph
    for idx, p in enumerate(partitions):
        subgraph_nodes = partitions[p]

        inbound_edges = set()
        outbound_edges = set()

        for u in subgraph_nodes:
            inbound_edges.update(set(u.input) - initializer_names)
            outbound_edges.update(set(u.output))

        # register inputs/outputs
        for tensor_idx, tensor_name in enumerate(inbound_edges):
            if tensor_name not in inputs_of: inputs_of[tensor_name] = []
            inputs_of[tensor_name].append(idx)

        for tensor_idx, tensor_name in enumerate(outbound_edges):
            if tensor_name not in outputs_of: outputs_of[tensor_name] = []
            outputs_of[tensor_name].append(idx)

    subgraph_inputs = {i: set() for i in range(len(partitions))}
    subgraph_outputs = {i: set() for i in range(len(partitions))}

    for tensor_name in inputs_of.keys() & outputs_of.keys():
        for psrc in outputs_of[tensor_name]:
            for pdst in inputs_of[tensor_name]:
                if psrc != pdst:
                    subgraph_outputs[psrc].add(tensor_name)
                    subgraph_inputs[pdst].add(tensor_name)

    for tensor_name in inputs_of.keys():
        if tensor_name in model_inputs:
            for pdst in inputs_of[tensor_name]:
                subgraph_inputs[pdst].add(tensor_name)

    for tensor_name in outputs_of.keys():
        if tensor_name in model_outputs:
            for psrc in outputs_of[tensor_name]:
                subgraph_outputs[psrc].add(tensor_name)

    inputs_of = dict()
    outputs_of = dict()

    for i in range(len(partitions)):
        subgraph_inputs[i] = list(subgraph_inputs[i])
        subgraph_outputs[i] = list(subgraph_outputs[i])

    # a tensor is an input to the graph if its destination is some subgraph node
    # but its source is not
    for idx, p in enumerate(partitions):
        extractor = onnx.utils.Extractor(model)
        subgraph_nodes = partitions[p]

        inputs = subgraph_inputs[idx]
        outputs = subgraph_outputs[idx]

        partition = extractor.extract_model(inputs, outputs)
        onnx_partitions.append(Partition(partition, partition.graph, inputs, outputs))

        # register inputs/outputs
        for tensor_idx, tensor_name in enumerate(inputs):
            if tensor_name not in inputs_of: inputs_of[tensor_name] = []
            inputs_of[tensor_name].append((idx, tensor_idx))

        for tensor_idx, tensor_name in enumerate(outputs):
            if tensor_name not in outputs_of: outputs_of[tensor_name] = []
            outputs_of[tensor_name].append((idx, tensor_idx))

    ps = PartitionSet(model, onnx_partitions)
    for tensor_name in outputs_of:
        if tensor_name not in inputs_of: continue
        for src, src_idx in outputs_of[tensor_name]:
            for dst, dst_idx in inputs_of[tensor_name]:
                ps.connect(src, dst, src_idx, dst_idx)

    return ps

def partition_graph(model, num_partitions, num_tries=20) -> PartitionSet:
    if num_partitions == 1: num_tries = 1
    best_ps = None

    for _ in range(num_tries):
        partition_specs = random_contraction(model.graph, num_partitions, approx_equal_sizes=True)
        ps = export_partitions_onnx(model, partition_specs)
        # print(_, ps.partition_sizes())

        if best_ps is None or ps.partition_sizes_std() < best_ps.partition_sizes_std():
            best_ps = ps

    print(best_ps.partition_sizes())
    return best_ps

def worker(model, num_partitions):
    seed = (getpid() ^ perf_counter_ns()) & ((1<<32)-1)
    random.seed(seed)
    np.random.seed(seed)
    partition_specs = random_contraction(model.graph, num_partitions, approx_equal_sizes=True)
    ps = export_partitions_onnx(model, partition_specs)
    print(getpid(), seed, ps.partition_sizes())
    return ps

def partition_graph_parallel(model, num_partitions, num_tries=10, num_threads=8) -> PartitionSet:
    with Pool(num_threads) as pool:
        results = pool.starmap(worker, [(model, num_partitions)] * num_tries)
        results = list(results)

        best_ps = None
        for ps in results:
            if best_ps is None or ps.partition_sizes_std() < best_ps.partition_sizes_std():
                best_ps = ps

    print("best", best_ps.partition_sizes())
    return best_ps
