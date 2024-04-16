
import argparse
import networkx as nx
import pandas as pd
import pickle
import onnx
import sys
import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from proteus.graph import Graph
from proteus.graph.onnx.partition import partition_graph, partition_graph_parallel
from onnx import helper, shape_inference
import time
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_dir", type=str)
parser.add_argument("--onnx_file", type=str, help="only need to specify if we're processing only a single file")
parser.add_argument("--outfile", type=str, default="real.pkl")
parser.add_argument("--nthreads", type=int, default=32)
parser.add_argument("--reps", type=int, default=3)
args = parser.parse_args()

# print graph sizes
if args.onnx_dir:
    pd_rows = []
    for w in glob(os.path.join(args.onnx_dir, "*.onnx")):
        model = onnx.load(w)
        pd_rows.append([w, len(model.graph.node)])
    print(pd.DataFrame.from_records(pd_rows, columns=["filename", "num_nodes"]))

def process_one(onnx_path):
    seed = os.getpid() ^ int(time.time())
    random.seed(seed)
    np.random.seed(seed)

    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    n_nodes = len(model.graph.node)
    n_partitions = n_nodes // 16
    n_tries = 30
    if n_nodes > 500: n_tries = 5
    ps = partition_graph(model, n_partitions, num_tries=n_tries)

    ret = []
    for p in range(n_partitions):
        G = Graph.from_onnx(ps.partitions[p].model)
        serialized = ps.partitions[p].model.SerializeToString()
        graph = G.make_networkx(model=None, fix_num_inputs=False)
        graph.graph.update({
            "onnx_file": onnx_path,
            "partition_id": p
        })
        ret.append((graph, serialized))

    return ret

if not args.onnx_file and not args.onnx_dir:
    print("Error. Must specify either onnx_file or onnx_dir")
    sys.exit(1)

onnx_files = []
if args.onnx_file:
    onnx_files = [ args.onnx_file ]
    results = list(map(process_one, onnx_files))

    with open(args.outfile, "wb") as fp:
        pickle.dump(results, fp)
else:
    onnx_files = list(glob(os.path.join(args.onnx_dir, "*.onnx")))
    onnx_files = onnx_files * args.reps

    with Pool(min(args.nthreads, len(onnx_files))) as pool:
        results = pool.imap_unordered(process_one, onnx_files) # noqa:
        results = list(tqdm(results, total=len(onnx_files)))

        with open(args.outfile, "wb") as fp:
            pickle.dump(results, fp)
