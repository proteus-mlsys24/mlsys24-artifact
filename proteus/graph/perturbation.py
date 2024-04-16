import random
import copy
from typing import List, Dict, Callable
from proteus.graph import Graph, Edge, FixedNode, Placeholder, OperatorRegistry, GraphNode


def proteus_node_from_onnx(node: FixedNode):
    if node.onnx_node.op_type in OperatorRegistry.operator_lookup:
        operator_cls = OperatorRegistry.operator_lookup[node.onnx_node.op_type]
        print(operator_cls)


def replace_node_with_placeholder(G: Graph, node: FixedNode):
    if node.onnx_node.op_type not in OperatorRegistry.operator_lookup:
        return

    node_idx = G.nodes.index(node)
    op_cls = OperatorRegistry.operator_lookup[node.onnx_node.op_type]
    placeholder = Placeholder(op_cls.num_inputs, max(1, G.node_num_outputs(node)), replacement_node=node)
    # placeholder = Placeholder(G.node_num_inputs(node), G.node_num_outputs(node))
    # print(f"op_cls {op_cls}, inputs: spec={op_cls.num_inputs}, actual={G.node_num_inputs(node)}, outputs: {G.node_num_outputs(node)}")

    G.nodes[node_idx] = placeholder
    for idx, e in enumerate(G.edges):
        if e.src == node:
            G.edges[idx] = Edge(placeholder, e.dst, e.srcIdx, e.dstIdx)
        elif e.dst == node:
            G.edges[idx] = Edge(e.src, placeholder, e.srcIdx, e.dstIdx)


def replace_nodes_with_placeholder(G: Graph, node_count: int):
    random.seed(1337)
    filter_fn: Callable[[GraphNode], bool] = lambda node: node.onnx_node.op_type in OperatorRegistry.operators
    eligible_nodes = list(filter(filter_fn, G.nodes))
    nodes = random.sample(eligible_nodes, node_count)
    for node in nodes:
        replace_node_with_placeholder(G, node)


def graph_split_vertex(G: Graph, node: GraphNode):
    num_inputs = G.node_num_inputs(node)
    num_outputs = G.node_num_outputs(node)
    u = Placeholder(G.node_num_inputs(node), 1)
    v = Placeholder(1, G.node_num_outputs(node))

    deleted_edges = []
    additional_edges = []

    # inputs of node -> inputs of u
    # outputs of node -> outputs of v
    for edge in G.edges:
        if edge.dst == node:
            # something -> node
            deleted_edges.append(edge)
            additional_edges.append(Edge(edge.src, u, edge.srcIdx, edge.dstIdx))
        elif edge.src == node:
            # node -> something
            deleted_edges.append(edge)
            additional_edges.append(Edge(v, edge.dst, edge.srcIdx, edge.dstIdx))

    # edge from u -> v
    additional_edges.append(Edge(u, v, 0, 0))

    # update nodes
    G.nodes.remove(node)
    G.nodes.append(u)
    G.nodes.append(v)

    # update edges
    for edge in deleted_edges:
        G.edges.remove(edge)
    G.edges.extend(additional_edges)

    assert G.node_num_inputs(u) == num_inputs
    assert G.node_num_outputs(v) == num_outputs


def graph_join_vertex(G: Graph, u: GraphNode, v: GraphNode):
    num_inputs = G.node_num_inputs(u)
    num_outputs = G.node_num_outputs(v)
    node = Placeholder(num_inputs, num_outputs)

    deleted_edges = []
    additional_edges = []
    for edge in G.edges:
        if edge.src == u and edge.dst == v:
            deleted_edges.append(edge)
        if edge.dst == u:
            deleted_edges.append(edge)
            additional_edges.append(Edge(edge.src, node, edge.srcIdx, edge.dstIdx))
        elif edge.src == v:
            deleted_edges.append(edge)
            additional_edges.append(Edge(node, edge.dst, edge.srcIdx, edge.dstIdx))

    # update nodes
    G.nodes.remove(u)
    G.nodes.remove(v)
    G.nodes.append(node)

    # update edges
    for edge in deleted_edges:
        G.edges.remove(edge)
    G.edges.extend(additional_edges)

    assert G.node_num_inputs(node) == num_inputs
    assert G.node_num_outputs(node) == num_outputs


def graph_pick_simple_edge(G: Graph):
    def is_simple(edge: Edge):
        return G.node_num_outputs(edge.src) == 1 and G.node_num_inputs(edge.dst) == 1

    simple_edges = list(filter(is_simple, G.edges))
    if len(simple_edges) > 0:
        return random.choice(simple_edges)


def graph_random_mutation(G: Graph):
    if random.random() < 0.5:
        # contraction
        edge = graph_pick_simple_edge(G)
        if edge: graph_join_vertex(G, edge.src, edge.dst)
    else:
        # splitting
        node = random.choice(G.nodes)
        graph_split_vertex(G, node)


def generate_topological_mutants(G: Graph, depth: int, count: int):
    current: List[Graph] = [G]
    next_round: List[Graph] = list()
    for round in range(depth):
        # print(f"round: {round}, current: {len(current)}")
        next_round = []
        for graph in current:
            for i in range(count):
                next_graph = copy.deepcopy(graph)
                graph_random_mutation(next_graph)
                next_round.append(next_graph)

        next_round = random.sample(next_round, count)
        # for idx, graph in enumerate(next_round):
        #     dot = graph.make_dot()
        #     dot.render(f"/tmp/mutant_round{round}_{idx}", cleanup=True)

        current = next_round

    return next_round


def convert_fixed_nodes_to_placeholders(G: Graph):
    converted: Dict[FixedNode, Placeholder] = dict()
    GG = Graph()
    cons = []

    for node in G.nodes:
        # create the replacement nodes
        replacement = Placeholder(G.node_num_inputs(node), max(1, G.node_num_outputs(node)), replacement_node=node)
        GG.nodes.append(replacement)
        converted[node] = replacement

        # add the constraints
        opcode_id = OperatorRegistry.operator_ids[node.onnx_node.op_type]
        cons.append(replacement.node_opcode == opcode_id)

    for edge in G.edges:
        GG.edges.append(Edge(
            converted[edge.src], converted[edge.dst],
            edge.srcIdx, edge.dstIdx
        ))

    return GG, cons
