
import random
import networkx as nx

def random_contraction(graph: nx.DiGraph, k, min_partition_size=None, approx_equal_sizes=False):
    par = dict()
    psize = dict()

    edges = list(set(graph.edges))
    outputs = dict()
    for node in graph.nodes:
        par[node] = node
        psize[node] = 1

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
    for node in graph.nodes:
        root = find_root(node)
        if root not in partitions: partitions[root] = list()
        partitions[root].append(node)

    # print(f"We got {len(partitions)} partitions.")
    return partitions