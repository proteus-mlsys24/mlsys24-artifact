
import random
import matplotlib.pyplot as plt
from proteus.graph.graphrnn.graphrnn_utils import load_and_process_graphrnn_graphs, draw_graph, ensure_num_inputs, ensure_num_inputs_connected, ensure_num_inputs_outputs_connected
from proteus.config import ProteusConfig

all_graphs = []
for path in ProteusConfig.graphrnn_graphs:
    graphs = load_and_process_graphrnn_graphs(path, is_real=False)
    all_graphs.extend(graphs)
    
    print(path, len(graphs))

for _ in range(16):
    graph = random.choice(all_graphs)
    # graph2 = ensure_num_inputs(graph, 2)
    # graph2 = ensure_num_inputs_connected(graph, 2)
    graph2 = ensure_num_inputs_outputs_connected(graph, 2, 3)
    if graph2:
        plt.figure()
        draw_graph(graph2)
        plt.show()