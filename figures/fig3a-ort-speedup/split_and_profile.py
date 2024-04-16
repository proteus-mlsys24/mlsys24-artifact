
import argparse
import os
from glob import glob
import math
import onnx
import random
import numpy as np
from tqdm import tqdm 
import termcolor

from proteus.graph.onnx.partition import PartitionSet, random_contraction, export_partitions_onnx
from proteus.graph.onnx.partition import partition_graph, partition_graph_parallel
from proteus.backends.ort import optimize_partitions, profile_one, optimize_graph
from proteus.graph.onnx.utils import replace_mem_instrs_with_identity

blacklist = ["vit"]

parser = argparse.ArgumentParser()
parser.add_argument("onnx_path", type=str, help="Directory from which to load ONNX files")
parser.add_argument("out_path", type=str, help="Directory to write files into")
args = parser.parse_args()


def measure_one(model_name, model, num_partitions):
    experiment_name = f"{model_name}+{num_partitions}"
    experiment_dir = os.path.join(args.out_path, experiment_name)

    if os.path.exists(experiment_dir):
        n_existing = len(glob(os.path.join(experiment_dir, "reassembled*.json")))
        if n_existing > 10:
            return

    os.makedirs(os.path.join(args.out_path, experiment_name), exist_ok=True)
    fp = open(os.path.join(args.out_path, experiment_name, "log"), "w")

    if num_partitions > 1:
        # Step 1a. Split the graph
        # ps = partition_graph(model, num_partitions, num_tries=7)
        ps = partition_graph_parallel(model, num_partitions, num_tries=8)
        fp.write(f"Partition Sizes: {ps.partition_sizes()}\n")

        # Step 1b. Reassemble and save the model to ensure functional correctness
        reassembled = ps.reassemble()
        reassembled_path = os.path.join(args.out_path, experiment_name, "reassembled.onnx")
        onnx.save(reassembled, reassembled_path)

        # Step 2a. Perform optimization on each of the partitions
        optimize_partitions(ps)

        # Step 2b. Reassemble the optimized model
        optimized = ps.reassemble()
        replace_mem_instrs_with_identity(optimized)
        optimized_path = os.path.join(args.out_path, experiment_name, "optimized.onnx")
        onnx.save(optimized, optimized_path)

        # Step 3. Measure both models
        profile_one(reassembled_path, "none", os.path.join(args.out_path, experiment_name, "reassembled"))
        profile_one(optimized_path, "none", os.path.join(args.out_path, experiment_name, "optimized"))

        # Setp 4. Cleanup
        os.remove(reassembled_path)
        os.remove(optimized_path)

        fp.close()
    else:
        reassembled_path = os.path.join(args.out_path, experiment_name, "reassembled.onnx")
        onnx.save(model, reassembled_path)

        optimized = optimize_graph(model)["all"]
        optimized_path = os.path.join(args.out_path, experiment_name, "optimized.onnx")
        onnx.save(optimized, optimized_path)

        profile_one(reassembled_path, "none", os.path.join(args.out_path, experiment_name, "reassembled"))
        profile_one(optimized_path, "none", os.path.join(args.out_path, experiment_name, "optimized"))

        fp.close()

def run_model(model_path, model):
    basename = os.path.basename(model_path)
    model_name = basename[:basename.find(".")]

    if any([x in model_name for x in blacklist]): return
    n_nodes = len(model.graph.node)
    nparts = [n_nodes // 16]
    print("Trying nparts=", nparts)
    for count in nparts:
        for _ in range(5):
            # measure_one(model_name, model, count)
            try:
                measure_one(model_name, model, count)
            except Exception as e:
                print("error", e)
                # raise e


onnx_files = list(glob(os.path.join(args.onnx_path, "*.onnx")))
print(onnx_files)

for idx, onnx_path in enumerate(onnx_files):
    print(termcolor.colored(f"[{idx+1} / {len(onnx_files)}]: Processing {onnx_path}", "yellow"))
    try:
        onnx.checker.check_model(onnx_path)

        model = onnx.load(onnx_path)
        if len(model.graph.node) > 2000: 
            print("Skipping model", os.path.basename(onnx_path))
            continue
        run_model(onnx_path, model)
    except Exception as e:
        print(e)
        # raise e
