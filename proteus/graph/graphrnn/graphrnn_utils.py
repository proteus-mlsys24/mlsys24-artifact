import queue
import random
import networkx as nx
from networkx.algorithms.shortest_paths.generic import has_path
from proteus.config import ProteusConfig


def bfs(graph, src):
    dist = {src: 0}

    # keep track of BFS order
    order = dict()
    clock = 0

    q = queue.Queue()
    q.put(src)
    while not q.empty():
        cur = q.get()
        order[cur] = clock
        clock += 1

        for n in graph.neighbors(cur):
            if n not in dist:
                dist[n] = dist[cur] + 1
                q.put(n)

    return dist, order


# returns one of the diameters of the graph (u, v)
def graph_diameter(graph):
    dist1, _ = bfs(graph, next(iter(graph.nodes.keys())))
    u = max(dist1, key=lambda n: dist1[n])
    dist2, _ = bfs(graph, u)
    v = max(dist2, key=lambda n: dist2[n])
    return u, v


def draw_graph(graph, diameter=tuple(), solution=None):
    color_default = "#FFFFFF"
    color_diameter = "#FFFFFF"
    node_list = list(graph.nodes)
    node_colors = [color_default if n not in diameter else color_diameter for n in node_list]
    if solution is None:
        color_default = "#1F2041"
        color_diameter = color_default
        # color_diameter = "#000000"
        node_colors = [color_default if n not in diameter else color_diameter for n in node_list]
        empty_labels = {n: "" for n in node_list}
        nx.draw_networkx(graph, nodelist=node_list, node_color=node_colors,
                         labels=empty_labels,
                         node_size=100,
                         pos=nx.nx_agraph.graphviz_layout(graph, prog="dot"))
    else:
        nx.draw_networkx(graph, nodelist=node_list, node_color=node_colors,
                         labels=solution,
                         pos=nx.nx_agraph.graphviz_layout(graph, prog="dot"),
                         font_size=20,
                         node_size=3000,
                         arrowsize=20,
                         node_shape="none",
                         bbox=dict(facecolor="skyblue"))


def induce_orientation(graph, src):
    _, order = bfs(graph, src)
    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes)
    for (u, v) in graph.edges:
        if order[u] > order[v]: u, v = v, u
        G.add_edge(u, v)

    return G


def prune_graph(graph, v):
    nodelist = list(graph.nodes.keys())
    to_prune = list()
    for node in nodelist:
        if not has_path(graph, node, v):
            to_prune.append(node)

    graph.remove_nodes_from(to_prune)


def get_num_inputs_outputs(G: nx.DiGraph):
    input_nodes = [node for node in G.nodes if G.in_degree(node) == 0]
    output_nodes = [node for node in G.nodes if G.out_degree(node) == 0]

    return len(input_nodes), len(output_nodes)


def prune_graph_new(G: nx.DiGraph, num_inputs: int, num_outputs: int):
    """
    Solves ths problem of the graph having too many input or output nodes.
    """
    G = G.copy()

    input_nodes = [node for node in G.nodes if G.in_degree(node) == 0]
    output_nodes = [node for node in G.nodes if G.out_degree(node) == 0]

    if (num_inputs < num_outputs and len(input_nodes) > len(output_nodes)) or \
            (num_inputs > num_outputs and len(output_nodes) > len(input_nodes)):
        G = G.reverse()
        input_nodes = [node for node in G.nodes if G.in_degree(node) == 0]
        output_nodes = [node for node in G.nodes if G.out_degree(node) == 0]

    original_in, original_out = len(input_nodes), len(output_nodes)

    # sample nodes
    if num_inputs < len(input_nodes): input_nodes = random.sample(input_nodes, k=num_inputs)
    if num_outputs < len(output_nodes): output_nodes = random.sample(output_nodes, k=num_outputs)

    # prune
    to_prune = []
    for node in G.nodes:
        input_path = any(has_path(G, i, node) for i in input_nodes)
        output_path = any(has_path(G, node, o) for o in output_nodes)

        if not (input_path and output_path):
            to_prune.append(node)

    G.remove_nodes_from(to_prune)

    # ins, outs = get_num_inputs_outputs(G)
    # print(f"in: {original_in} -> {ins} (req {num_inputs}), out: {original_out} -> {outs} (req {num_outputs})")

    return G


def process_batch(graphs):
    ret = []

    for graph in graphs:
        u, v = graph_diameter(graph)
        u, v = v, u
        directed = induce_orientation(graph, u)
        # prune_graph(directed, v)

        ret.append(directed)

    return ret


def ensure_num_inputs(graph: nx.DiGraph, num_inputs: int):
    graph = graph.copy()
    while len(graph.nodes) > 0:
        input_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]  # noqa
        if len(input_nodes) >= num_inputs: return graph

        victim = random.choice(input_nodes)
        graph.remove_node(victim)

        while True:
            changed = False
            for node in graph.nodes:
                if graph.in_degree(node) == 0 and graph.out_degree(node) == 0:  # noqa
                    graph.remove_node(node)
                    changed = True
                    break
            if not changed: break


def ensure_num_inputs_connected(graph: nx.DiGraph, num_inputs: int):
    count = 0
    while count < 100:
        count += 1
        graph2 = ensure_num_inputs(graph, num_inputs)
        if graph2 and nx.is_connected(graph2.to_undirected()):
            return graph2


def ensure_num_inputs_outputs_connected(graph: nx.DiGraph, num_inputs: int, num_outputs: int):
    graph2 = ensure_num_inputs_connected(graph, num_inputs)
    if graph2:
        graph3 = ensure_num_inputs_connected(graph2.reverse(), num_outputs)
        if graph3:
            return prune_graph_new(graph3.reverse(), num_inputs, num_outputs)


def load_and_process_graphrnn_graphs(graph_path, is_real):
    import sys
    sys.path.append(ProteusConfig.graphrnn_path)
    from utils import load_graph_list  # noqa

    graphs = load_graph_list(graph_path, is_real=is_real)
    return process_batch(graphs)
