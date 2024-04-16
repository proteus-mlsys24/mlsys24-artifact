
import argparse
from glob import glob
import os
import pickle
from proteus.graph.graphrnn.sampling_new import TopologyDistribution
from proteus.graph.graphrnn.graphrnn_utils import load_and_process_graphrnn_graphs
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("graphrnn_path", type=str)
parser.add_argument("--nthreads", type=int, default=32)
args = parser.parse_args()

graph_paths = list(glob(os.path.join(args.graphrnn_path, "*.dat")))

graphs = []
for graph_path in graph_paths:
    graphs.extend(load_and_process_graphrnn_graphs(graph_path, False))

graph_degrees = {(1, 1)}
for in_deg in range(1, 5):
    for out_deg in range(1, 5):
        graph_degrees.add((in_deg, out_deg))


def do_one(input_nodes, output_nodes):
    try:
        filename = f"distrib_cache/distrib_{input_nodes}_{output_nodes}.pkl"
        if os.path.exists(filename): 
            print("Skipped.", input_nodes, output_nodes)
            return
        distrib = TopologyDistribution(graphs, input_nodes, output_nodes, num_samples=5)
        with open(filename, "wb") as fp:
            pickle.dump(distrib, fp)
        print("Done!", input_nodes, output_nodes)
    except Exception as e:
        print("Failed.", input_nodes, output_nodes, e)

os.makedirs("distrib_cache", exist_ok=True)
with Pool(args.nthreads, maxtasksperchild=1) as pool:
    pool.starmap(do_one, graph_degrees)
