
import argparse
import torch
import onnx
import networkx as nx
import torchvision.models as models
import matplotlib.pyplot as plt

import os
import sys
import copy
import itertools
import random
import warnings
from typing import Dict, Set, Tuple, List
from termcolor import colored

import hidet
hidet.option.search_space(2)
from hidet.graph.frontend.torch.utils import serialize_output, symbol_like_torch

import numpy as np

from random_contraction import random_contraction

from hidet.utils import benchmark_func

def bench_hidet_graph(graph: hidet.FlowGraph, graph_inputs):
    cuda_graph = graph.cuda_graph()
    output = cuda_graph.run(graph_inputs)
    return benchmark_func(lambda: cuda_graph.run())


class HidetPartitionSet:
    def __init__(self, graph, partitions):
        self.graph = graph
        self.partitions = partitions

        self.tensor_lookup: Dict[hidet.Tensor, hidet.Tensor] = dict()
        
        # collect all graph output tensors
        self.node_outputs: Set[hidet.Tensor] = set()
        for node in self.graph._nodes:
            self.node_outputs.update(node.outputs)

        # collect the global set of constant inputs
        self.global_constant_inputs: Set[hidet.Tensor] = set()
        node_outputs: Set[hidet.Tensor] = set()
        for node in self.graph.nodes:
            node_outputs.update(node.outputs)
        for node in self.graph.nodes:
            for i in node.inputs:
                if i not in self.graph.inputs and i not in node_outputs:
                    self.global_constant_inputs.add(i)
        print("Collected", len(self.global_constant_inputs), "constant inputs.")

        # collect inputs
        self.inputs_of: Dict[hidet.Tensor, List[hidet.Operator]] = dict()
        for node in self.graph.nodes:
            for t in node.inputs:
                if t not in self.inputs_of:
                    self.inputs_of[t]: List[hidet.Operator] = list()
                self.inputs_of[t].append(node)

    def partition_sizes(self):
        return list(map(len, self.partitions))

    def partition_sizes_std(self):
        return np.std(self.partition_sizes())

    def partition_edges(self):
        N = len(self.graph._nodes)

        output_of: Dict[hidet.Tensor, int] = dict()
        for idx in range(N):
            node = self.graph._nodes[idx]
            for o in node.outputs:
                output_of[o] = idx

        partition_of: Dict[int, int] = dict()
        for idx, p in enumerate(self.partitions):
            for n in p: partition_of[n] = idx

        # collect all inter-partition edges. 
        # elements are in the form of (src node, dst node, output id, input id)
        partition_outs = [ list() for idx in range(len(self.partitions)) ]
        partition_ins  = [ list() for idx in range(len(self.partitions)) ]
        
        for pid, pnodes in enumerate(self.partitions):
            for nidx in pnodes:
                for i, ti in enumerate(self.graph._nodes[nidx].inputs):
                    if ti not in output_of:
                        partition_ins[pid].append(ti)

                for o, to in enumerate(self.graph._nodes[nidx].outputs):
                    if to not in output_of:
                        partition_outs[pid].append(to)

        for a in range(N):
            for b in range(a+1, N):
                if partition_of[a] == partition_of[b]: continue
                for o, to in enumerate(self.graph._nodes[a].outputs):
                    for i, ti in enumerate(self.graph._nodes[b].inputs):
                        if ti == to:
                            psrc, pdst = partition_of[a], partition_of[b]
                            partition_outs[psrc].append(ti)
                            partition_ins[pdst].append(ti)

        return partition_outs, partition_ins

    def extract_partition_new(self, pid: int):
        # create the new nodes
        original_nodes = []
        new_nodes = []
        all_inputs: Set[hidet.Tensor] = set()
        all_outputs: Set[hidet.Tensor] = set()
        
        for idx in self.partitions[pid]:
            node = copy.copy(self.graph._nodes[idx])
            original_nodes.append(self.graph._nodes[idx])
            new_nodes.append(node)

        # duplicate (and register globally) the tensors
        all_tensors: Set[hidet.Tensor] = set()
        for node in new_nodes:
            all_tensors.update(node.inputs)
            all_tensors.update(node.outputs)

        tensor_replacements: Dict[hidet.Tensor, hidet.Tensor] = {
            t: copy.copy(t) for t in all_tensors
        }

        inv_tensor_replacements: Dict[hidet.Tensor, hidet.Tensor] = {
            v: k for k,v in tensor_replacements.items()
        }

        for original, copied in tensor_replacements.items():
            if copied in self.tensor_lookup:
                raise Exception("tensor already found")
            self.tensor_lookup[copied] = original

        for node in new_nodes:
            for idx, t in enumerate(node.inputs):
                node.inputs[idx] = tensor_replacements[t]
            for idx, t in enumerate(node.outputs):
                node.outputs[idx] = tensor_replacements[t]

        # find the subgraph inputs and outputs
        for node in new_nodes:
            all_outputs.update(node.outputs)
            all_inputs.update(node.inputs)

        internal_tensors: Set[hidet.Tensor] = set()
        for node in new_nodes:
            for i in node.inputs:
                if i in all_outputs:
                    internal_tensors.add(i)

        partition_inputs = list(all_inputs - internal_tensors)

        partition_outputs = []
        for node in new_nodes:
            for output_tensor in node.outputs:
                original_output = inv_tensor_replacements[output_tensor]
                if original_output not in self.inputs_of or \
                        any(node not in original_nodes for node in self.inputs_of[original_output]):
                    partition_outputs.append(output_tensor)

        constant_inputs = {
            t for t in partition_inputs
            if self.tensor_lookup[t] in self.global_constant_inputs
        }

        print(f"This partition has {len(partition_inputs)} inputs (including {len(constant_inputs)} constants), and {len(new_nodes)} nodes.")

        # disconnect the inputs to ensure graph tracing works
        for i in partition_inputs:
            if i not in constant_inputs:
                i._storage = None
                pass
            i._op = None
            i._trace = None
            # i._storage = None
        
        non_constant_partition_inputs = [
                i for i in partition_inputs if i not in constant_inputs 
        ]

        # finally return the new subgraph
        subgraph = hidet.FlowGraph(partition_outputs, non_constant_partition_inputs, new_nodes)
        return subgraph

    def get_edges_between_partitions(self, partitions):
        edges = []
        nparts = len(partitions)
        for a in range(nparts):
            for b in range(nparts):
                if a == b: continue
                for idx1, t1 in enumerate(partitions[a].outputs):
                    for idx2, t2 in enumerate(partitions[b].inputs):
                        tt1 = best_ps.tensor_lookup[t1]
                        tt2 = best_ps.tensor_lookup[t2]
                        if tt1 == tt2:
                            edges.append((a, b, idx1, idx2))
                            print(a, b, idx1, idx2, colored(tt1.signature(), "yellow"))
                            
        return edges
    
    def match_module_inputs_outputs(self, graphs):
        graph_inputs = [ None for _ in range(len(graph.inputs)) ]
        graph_outputs = [ None for _ in range(len(graph.outputs)) ]

        for idx, subgraph in enumerate(graphs):
            # find inputs
            for subgraph_input_idx, subgraph_input in enumerate(subgraph.inputs):
                for graph_input_idx, graph_input in enumerate(self.graph.inputs):
                    if id(subgraph_input) == id(graph_input):
                        graph_inputs[graph_input_idx] = (idx, subgraph_input_idx)

            # find outputs
            for subgraph_output_idx, subgraph_output in enumerate(subgraph.outputs):
                for graph_output_idx, graph_output in enumerate(self.graph.inputs):
                    if id(subgraph_output) == id(graph_output):
                        graph_outputs[graph_output_idx] = (idx, subgraph_output_idx)

        print("graph inputs", graph_inputs)
        print("graph outputs", graph_outputs)

        return graph_inputs, graph_outputs
    
    def validate_partitions(self, graphs, graphs_opt):
        for idx, (A, B) in enumerate(zip(graphs, graphs_opt)):
            assert len(A.inputs) == len(B.inputs), f"partition {idx}: num inputs must match"
            assert len(A.outputs) == len(B.outputs), f"partition {idx}: num outputs must match"

            print(f"partition {idx} has {len(A.inputs)} inputs and {len(A.outputs)} outputs")


    def reassemble_partitions(self, graphs, graphs_opt=None):
        self.validate_partitions(graphs, graphs_opt)
        self.match_module_inputs_outputs(graphs)

        nparts = len(partitions)
        edges = self.get_edges_between_partitions(graphs)

        # 0: src pid, 1: dst pid, 2: output id, 3: input id
        edges_lookup = { (e[1], e[3]): (e[0], e[2]) for e in edges }
        print(edges_lookup)
        for srcpid, dstpid, outputid, inputid in edges:
            print(f"input {inputid} (total: {len(graphs_opt[dstpid].inputs)}) of partition {dstpid} is output {outputid} (total: {len(graphs_opt[srcpid].outputs)}) of partition {srcpid}")

        # maintain the output tensors from previous partitions

        # remove all the claimed inputs at the end so we don't mutate
        # the subgraphs incorrectly
        inputs_to_remove: List[Tuple[int, hidet.Tensor]] = list()

        # maps the "input" tensors of the subsequent graph to the "correct" output of the previous graph
        # we first collect these tensors, then go into the subsequent graph and replaces all the corresponding inputs
        # and finally we should remove all "inputs" of the subsequent graph that are "claimed" by this mapping
        claimed_inputs: Dict[hidet.Tensor, hidet.Tensor] = dict()
        for srcpid, dstpid, outputid, inputid in edges:
            i = graphs_opt[dstpid].inputs[inputid]
            o = graphs_opt[srcpid].outputs[outputid]
            claimed_inputs[i] = o

        # for pidx in partition_order:
        for pidx in range(nparts):
            n_inputs = len(graphs_opt[pidx].inputs)

            """
            for input_idx in range(n_inputs):
                if (pidx, input_idx) in edges_lookup:
                    source_idx, source_output_idx = edges_lookup[(pidx, input_idx)]
                    i = graphs_opt[pidx].inputs[input_idx]
                    o = graphs_opt[source_idx].outputs[source_output_idx]
                    claimed_inputs[i] = o
            """

            for node in graphs_opt[pidx].nodes:
                for input_idx in range(len(node.inputs)):
                    input_tensor = node.inputs[input_idx]
                    if input_tensor in claimed_inputs:
                        node.inputs[input_idx] = claimed_inputs[input_tensor]
                        print(pidx, id(claimed_inputs[input_tensor]))

            """
            for input_tensor in claimed_inputs:
                # graphs_opt[pidx].inputs.remove(input_tensor)
                inputs_to_remove.append((pidx, input_tensor))
            """

        # now remove all the claimed inputs
        for srcpid, dstpid, outputid, inputid in edges:
            inputs_to_remove.append((dstpid, graphs_opt[dstpid].inputs[inputid]))
        print(set(inputs_to_remove))

        for pidx, input_tensor in inputs_to_remove:
            graphs_opt[pidx].inputs.remove(input_tensor)

        graph_inputs: List[hidet.Tensor] = list()
        for pidx in range(nparts):
            print("partition", pidx, "num_inputs", len(graphs_opt[pidx].inputs))
            graph_inputs.extend(graphs_opt[pidx].inputs)

        graph_outputs = []
        for pid in range(nparts):
            for output_tensor in graphs_opt[pid].outputs:
                # if output_tensor not in not_outputs:
                if output_tensor not in claimed_inputs.values():
                    print("Found output from partition", pid, repr(output_tensor), id(output_tensor))
                    graph_outputs.append(output_tensor)

        assert len(graph_outputs) == 1, f"expected 1 graph output, got {len(graph_outputs)}"

        if len(graph_inputs) > 1:
            warnings.warn(f"The graph has {len(graph_inputs)} inputs instead of the usual 1.")

        merged_graph = hidet.FlowGraph(graph_outputs, inputs=graph_inputs)
        return merged_graph
    
def get_hidet_graph_for(model_name: str):
    model = models.get_model(model_name).eval().cuda() # torch.nn.Module

    interpreter = hidet.graph.frontend.from_torch(model) # hidet.graph.frontend.torch.interpreter.Interpreter
    symbolic_inputs = [ symbol_like_torch(torch.rand(8, 3, 224, 224).cuda()) ]
    output = interpreter(*symbolic_inputs)
    output_format, output_tensors = serialize_output(output)
    graph: hidet.FlowGraph = hidet.trace_from(output_tensors, inputs=symbolic_inputs)

    return graph

def get_hidet_graph_from_onnx(onnx_path: str):
    module = hidet.graph.frontend.from_onnx(onnx_path)

    input_shapes: Dict[str, Tuple[int, ...]] = dict()
    onnx_model = onnx.load_model(onnx_path)
    for input_elem in onnx_model.graph.input:
        input_name = input_elem.name
        input_shape = tuple(d.dim_value for d in input_elem.type.tensor_type.shape.dim)

        input_shapes[input_name] = input_shape

    placeholder: List[hidet.Tensor] = []
    for input_name in module.input_names:
        input_shape = input_shapes[input_name]
        data = hidet.symbol(input_shape, device='cuda', dtype='int32')
        placeholder.append(data)

    output = module(*placeholder)
    graph = hidet.trace_from(output, inputs=placeholder)
    return graph

def hidet_graph_to_networkx(graph: hidet.FlowGraph):
    G = nx.DiGraph()
    # G.add_nodes_from(range(len(graph._nodes)))
    N = len(graph._nodes)
    for i in range(N):
        G.add_node(i, signature=str(graph._nodes[i]))
    edges = []
    for a, node_a in enumerate(graph._nodes):
        for b, node_b in enumerate(graph._nodes):
            for idx1, t1 in enumerate(node_a.outputs):
                for idx2, t2 in enumerate(node_b.inputs):
                    if t1 == t2:
                        edges.append((a, b, idx1, idx2))
                        G.add_edge(a, b)

    return G, edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--onnx", type=str)
    parser.add_argument("--nparts", type=int)
    parser.add_argument("--partsize", type=int)
    parser.add_argument("--outfile", type=str)
    args = parser.parse_args()

    if args.model:
        graph = get_hidet_graph_for(args.model)
    elif args.onnx:
        graph = get_hidet_graph_from_onnx(args.onnx)
    else:
        raise Exception("Must specify either model or onnx")

    # graph = get_hidet_graph_for("alexnet")
    G, edges = hidet_graph_to_networkx(graph)
    print(f"Original graph has {len(G.nodes)} vertices.")

    nparts = 0
    if args.nparts is not None:
        nparts = args.nparts
    if args.partsize is not None:
        nparts = len(G.nodes) // args.partsize

    print("NPARTS", nparts)

    with hidet.graph.PassContext() as ctx:
        graph_opt = hidet.graph.optimize(graph)

    # warmup
    for _ in range(5):
        _ = bench_hidet_graph(graph, graph_inputs)
        _ = bench_hidet_graph(graph_opt, graph_inputs)

    graph_inputs = graph.dummy_inputs()
    time_original = bench_hidet_graph(graph, graph_inputs)
    time_optimized = bench_hidet_graph(graph_opt, graph_inputs)
    if nparts == 0:
        time_optimized = bench_hidet_graph(graph_opt, graph_inputs)
        print(f"INITIAL original: {time_original}, optimized: {time_optimized}, speedup: {time_original/time_optimized}")

        sys.exit(0)
    
    best_ps = None
    
    print("-" * 50)
    print("Original Graph:")
    print(colored(graph, "yellow"))
    print("-" * 50)

    for _ in range(50):
        partitions = random_contraction(G, nparts)
        ps = HidetPartitionSet(graph, list(partitions.values()))
        # print(_, ps.partition_sizes(), ps.partition_sizes_std())

        if best_ps is None or ps.partition_sizes_std() < best_ps.partition_sizes_std():
            best_ps = ps

    partitions = []
    partitions_opt = []

    original_total = 0
    optimized_total = 0

    for idx in range(nparts):
        print("-" * 50)
        print(f"idx={idx}")
        graph = best_ps.extract_partition_new(idx)
        print(colored(graph, "red"))
        with hidet.graph.PassContext() as ctx:
            graph_opt = hidet.graph.optimize(graph)
        print(colored(graph_opt, "green"))
        print("-" * 50)
        print()

        partitions.append(graph)
        partitions_opt.append(graph_opt)
        continue

    print("-" * 50)
    merged_graph = best_ps.reassemble_partitions(partitions, partitions_opt)
    print(merged_graph)
    print("-" * 50)

    for _ in range(5):
        _ = bench_hidet_graph(merged_graph, graph_inputs)

    graph_inputs = merged_graph.dummy_inputs()
    time_merged = bench_hidet_graph(merged_graph, graph_inputs)
    print("RESULT", time_original, time_optimized, time_merged)

    if args.outfile is not None:
        with open(args.outfile, "a") as fp:
            fp.write(f"{args.model if args.model is not None else args.onnx},{nparts},{time_original},{time_optimized},{time_merged}\n")
