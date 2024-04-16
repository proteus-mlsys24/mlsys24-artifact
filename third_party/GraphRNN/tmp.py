import utils
import networkx as nx
import matplotlib.pyplot as plt

graphs = utils.load_graph_list("graphs/GraphRNN_RNN_enzymes_4_128_test_0.dat", is_real=False)
print("loaded graphs:", len(graphs))

