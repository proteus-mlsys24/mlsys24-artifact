from typing import Dict, Union, List, Optional

import sys
import z3

from .onnx.utils import get_tensor_shapes

import onnx
import graphviz
import networkx as nx

from proteus.graph.nodes import GraphNode, Placeholder, FixedNode


class Edge:
    def __init__(self, src: GraphNode, dst: GraphNode, srcIdx: int, dstIdx: int):
        self.src = src
        self.dst = dst
        self.srcIdx = srcIdx
        self.dstIdx = dstIdx

    def __str__(self):
        return f"{self.src.label()} ({self.srcIdx}) => {self.dst.label()} ({self.dstIdx})"

    def constraints(self):
        # ignore if both ends are fixed nodes
        if isinstance(self.src, FixedNode) and isinstance(self.dst, FixedNode):
            return list()

        cons = []
        if isinstance(self.src, Placeholder) and isinstance(self.dst, Placeholder):
            for dim in range(Placeholder.max_dims):
                cons.append(self.src.output_dims[self.srcIdx][dim] == self.dst.input_dims[self.dstIdx][dim])

        elif isinstance(self.src, Placeholder) and isinstance(self.dst, FixedNode):
            expected_shape = self.dst.input_shapes[self.dstIdx]
            while len(expected_shape) < Placeholder.max_dims: expected_shape.insert(0, 0)

            for dim in range(Placeholder.max_dims):
                cons.append(self.src.output_dims[self.srcIdx][dim] == expected_shape[dim])

        elif isinstance(self.src, FixedNode) and isinstance(self.dst, Placeholder):
            expected_shape = self.src.output_shape
            while len(expected_shape) < Placeholder.max_dims: expected_shape.insert(0, 0)

            for dim in range(Placeholder.max_dims):
                cons.append(expected_shape[dim] == self.dst.input_dims[self.dstIdx][dim])

        return cons


class Graph:
    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.edges: List[Edge] = []

    @classmethod
    def from_onnx(cls, onnx_model: Union[onnx.ModelProto, str]):
        from .ops import OperatorRegistry, OpBase
        model = onnx_model
        if isinstance(model, str):
            model = onnx.load_model(model)

        model = onnx.shape_inference.infer_shapes(model)
        model_inputs = [i.name for i in model.graph.input]
        tensor_shapes = get_tensor_shapes(model)

        G = Graph()

        output_lookup = dict()

        for node in model.graph.node:
            if node.op_type == "Identity": continue

            input_shapes = [
                list(tensor_shapes[i]) if i in tensor_shapes else None
                for i in node.input
            ]
            output_shape = tensor_shapes[node.output[0]]

            fixed = FixedNode(node,
                              input_shapes=input_shapes, output_shape=output_shape)

            G.nodes.append(fixed)
            for idx, output_name in enumerate(node.output):
                assert output_name not in output_lookup, f"duplicate output name {output_name} found."
                output_lookup[output_name] = (fixed, idx)

        inputs = set()
        for dst in G.nodes:
            for dstIdx, input_name in enumerate(dst.onnx_node.input):
                inputs.add(input_name)
                if input_name not in output_lookup:
                    if input_name not in model_inputs:
                        # print(f"{input_name} not found")
                        pass
                    else:
                        # print(f"{input_name} is a model input")
                        pass
                    continue

                src, srcIdx = output_lookup[input_name]

                max_input = 1e9
                if dst.onnx_node.op_type in OperatorRegistry.operator_lookup:
                    op_cls = OperatorRegistry.operator_lookup[dst.onnx_node.op_type]
                    max_input = op_cls.num_inputs

                if dstIdx < max_input:
                    G.edges.append(Edge(src, dst, srcIdx, dstIdx))

        return G

    def node_num_inputs(self, node):
        assert node in self.nodes
        return len(list(filter(lambda e: e.dst == node, self.edges)))

    def node_num_outputs(self, node):
        assert node in self.nodes
        return len(list(filter(lambda e: e.src == node, self.edges)))

    def graph_input_nodes(self):
        input_nodes = set(self.nodes)
        for e in self.edges:
            if e.dst in input_nodes:
                input_nodes.remove(e.dst)
        return input_nodes

    def graph_output_nodes(self):
        output_nodes = set(self.nodes)
        for e in self.edges:
            if e.src in output_nodes:
                output_nodes.remove(e.src)
        return output_nodes

    def make_dot(self, model: z3.ModelRef = None):
        dot = graphviz.Digraph()

        node_lookup = dict()
        for idx, node in enumerate(self.nodes):
            node_lookup[node.node_name] = str(idx)
            if isinstance(node, Placeholder):
                dot.node(str(idx), node.label(model), shape="rectangle", color='red')
            else:
                dot.node(str(idx), node.label(), shape="rectangle")

        for edge in self.edges:
            u, v = edge.src.node_name, edge.dst.node_name
            if u not in node_lookup or v not in node_lookup:
                # print(f"Edge {u} -> {v} not found.")
                continue
            idx1 = node_lookup[u]
            idx2 = node_lookup[v]
            dot.edge(idx1, idx2, label=str(edge.dstIdx))

        return dot

    def make_networkx(self, model, fix_num_inputs=False):
        from proteus.graph import OperatorRegistry
        G = nx.DiGraph()
        node_ids = {node:idx for idx, node in enumerate(self.nodes)}

        for idx, node in enumerate(self.nodes):
            if isinstance(node, FixedNode):
                opcode = node.onnx_node.op_type
                max_inputs = None
                if fix_num_inputs and opcode in OperatorRegistry.operator_lookup:
                    max_inputs = OperatorRegistry.operator_lookup[opcode].num_inputs

                node_attrs = {
                    "input_shapes": node.input_shapes if max_inputs is None else node.input_shapes[:max_inputs],
                    "output_shape": node.output_shape,
                    **node.get_attr_dict()
                }
                G.add_node(idx, **node_attrs)
                # G.add_node(idx)
            else:
                if model is None:
                    raise Exception("Graph contains placeholders but model is not provided.")
                G.add_node(idx, **node.get_attr_dict(model))
                # G.add_node(idx)

        edges = [(node_ids[e.src], node_ids[e.dst]) for e in self.edges]
        G.add_edges_from(edges)
        return G

    def constraints(self):
        cons = []

        # node constraints
        for node in self.nodes:
            if isinstance(node, Placeholder):
                cons.extend(node.constraints())

        # edge constraints
        for edge in self.edges:
            edge_cons = edge.constraints()
            # if len(edge_cons) > 0:
            #     print("edge", edge.constraints())

            cons.extend(edge_cons)

        return cons

    def num_placeholders(self):
        return len(list(filter(lambda x: isinstance(x, Placeholder), self.nodes)))

    def reset_edge_indices(self):
        # input_edges: a list of edges that are incident to some node
        input_edges: Dict[GraphNode, List[Edge]] = dict()
        output_edges: Dict[GraphNode, List[Edge]] = dict()

        for e in self.edges:
            if e.dst not in input_edges: input_edges[e.dst] = list()
            input_edges[e.dst].append(e)

            if e.src not in output_edges: output_edges[e.src] = list()
            output_edges[e.src].append(e)

        # collect indices
        input_indices: Dict[GraphNode, Dict[Edge, int]] = dict()
        output_indices: Dict[GraphNode, Dict[Edge, int]] = dict()

        for node in input_edges:
            input_indices[node] = {e: idx for idx, e in enumerate(input_edges[node])}

        for node in output_edges:
            output_indices[node] = {e: idx for idx, e in enumerate(output_edges[node])}

        for edge in self.edges:
            edge.srcIdx = output_indices[edge.src][edge]
            edge.dstIdx = input_indices[edge.dst][edge]