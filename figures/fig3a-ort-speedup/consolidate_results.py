
import argparse
import os
import json
import onnx
import numpy as np
import pickle
from glob import glob
from scipy.stats import gmean
from tqdm import tqdm
import multiprocessing

def mean_runtime(filename):
    try:
        with open(filename, "r") as fp:
            trace = json.load(fp)

        runtimes = []
        for event in trace:
            if event["name"] == "SequentialExecutor::Execute":
                runtimes.append(event["dur"])

        if len(runtimes) == 0: return np.nan

        p95 = np.percentile(runtimes, 95)
        runtimes = list(filter(lambda x: x < p95, runtimes))
        mean = gmean(runtimes)

        return mean
    except:
        return np.nan

def mean_across_measurements(profile_dir, prefix):
    runtimes = []
    for f in glob(os.path.join(profile_dir, f"{prefix}*.json")):
        r = mean_runtime(f)
        if not np.isnan(r):
            runtimes.append(r)
        
    return gmean(runtimes)

def do_one(measurement_dir):
    measurement_name = os.path.basename(measurement_dir).split("+")[0]
    num_partitions = int(os.path.basename(measurement_dir).split("+")[1])
    mean_unoptimized = mean_across_measurements(measurement_dir, "reassembled")
    mean_optimized = mean_across_measurements(measurement_dir, "optimized")

    print(measurement_name, num_partitions, mean_unoptimized, mean_optimized)
    return ((measurement_name, num_partitions, mean_unoptimized, mean_optimized, mean_unoptimized / mean_optimized))


parser = argparse.ArgumentParser()
parser.add_argument("result_dirs", type=str, nargs='+')
args = parser.parse_args()

files = []

for f in args.result_dirs:
    tmp = list(sorted(glob(os.path.join(f, "*"))))
    files.extend(tmp)

with multiprocessing.Pool(16) as pool:
    output = list(pool.imap(do_one, tqdm(files, leave=False)))

for e in sorted(output):
    print(e)

with open("slowdown_data.pkl", "wb") as fp:
    pickle.dump(output, fp)