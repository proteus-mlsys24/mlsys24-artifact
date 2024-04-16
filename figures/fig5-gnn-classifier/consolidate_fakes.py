
from glob import glob
import random
import pickle
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--percentile", type=float, default=0.9, help="minimum logprob percentile from each group")
parser.add_argument("--count", type=int, default=1, help="number of graphs to sample from each group")
parser.add_argument("--outfile", type=str, default="consolidated_fakes.pkl")
parser.add_argument("--filter", type=str)
args = parser.parse_args()

files = list(filter(lambda fn: "trash" not in fn, glob("**/solutions.pkl", recursive=True)))
if args.filter:
    files = list(filter(lambda fn: args.filter in fn, files))

graphs = []

file_iter = tqdm(files, leave=False)
for f in file_iter:
    with open(f, "rb") as fp:
        tmp_graphs = pickle.load(fp)
        logprobs = list(map(lambda graph: graph.graph["logprob"], tmp_graphs))
        min_logprob = np.percentile(logprobs, int(args.percentile * 100))
        tmp_graphs = list(filter(lambda graph: graph.graph["logprob"] >= min_logprob, tmp_graphs))
        selected = random.choices(tmp_graphs, k=args.count)

    graphs.extend(selected)
    file_iter.write(f"{f}: count {len(selected)} total {len(graphs)}")

with open(args.outfile, "wb") as fp:
    pickle.dump(graphs, fp)