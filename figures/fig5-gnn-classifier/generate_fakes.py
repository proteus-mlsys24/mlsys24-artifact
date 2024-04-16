import os
import sys
import pickle
import time
import math
import itertools
import onnx
import traceback
import random
import argparse
import json
import signal
from typing import List
from multiprocessing import Pool

import networkx as nx
import matplotlib.pyplot as plt
from termcolor import colored
from tqdm import tqdm

from proteus.graph import Graph, OperatorRegistry
from proteus.graph.solver import get_random_initial_constraints, enumerate_solutions, SerializableModel
from proteus.graph.statistics.manager import SolutionManager
from proteus.graph.graphrnn.selector import ImportanceSamplingTopologySelector
from proteus.graph.heuristics.heur_channel_count import get_networkx_channel_counts, apply_heuristic
from proteus.graph.heuristics.heur_filter_opcode import collect_opcodes_networkx, apply_heuristic as apply_opcode_heuristic

from proteus.graph.perturbation import replace_nodes_with_placeholder, generate_topological_mutants, \
    graph_random_mutation

parser = argparse.ArgumentParser()
parser.add_argument("--real_pkl", type=str, default="real.pkl")
parser.add_argument("--num_mutants", type=int, default=0)
parser.add_argument("--no-random-enumeration", action="store_true")
parser.add_argument("--nthreads", type=int, default=16)
parser.add_argument("--timeout", type=int, default=5*60)
args = parser.parse_args()

attempts = 5
random_enumeration = not args.no_random_enumeration

generate_new_topology = args.num_mutants == 0
num_mutants = args.num_mutants

outdir = f"generated_{int(time.time())}"
os.makedirs(outdir)

def fix_num_inputs(graph: nx.DiGraph):
    for node in graph.nodes:
        opcode = graph.nodes[node]["opcode"]
        if opcode in OperatorRegistry.operator_lookup:
            max_inputs = OperatorRegistry.operator_lookup[opcode].num_inputs
            graph.nodes[node]["input_shapes"] = graph.nodes[node]["input_shapes"][:max_inputs]

def timeout_handler(s, f):
    raise TimeoutError

def thread_find_mutants_graphrnn(thread_name: str, real_topology: nx.DiGraph, serialized_model: bytes, max_solutions: int):
    time_begin = time.time()

    model = onnx.ModelProto.FromString(serialized_model)
    model_channel_counts = get_networkx_channel_counts(real_topology)
    model_opcodes = collect_opcodes_networkx(real_topology)

    # allow some non-model opcodes
    other_opcodes = random.sample(OperatorRegistry.operators, math.ceil(0.2 * len(model_opcodes)))
    model_opcodes.update(other_opcodes)

    fix_num_inputs(real_topology)
    initial_constraints = []

    original_dot = None

    # --------------------- Topology Sampling ------------------------------------------------

    if generate_new_topology:
        selector = ImportanceSamplingTopologySelector()
        good = False
        for attempt in range(attempts):
            try:
                topology, G, initial_constraints = selector.convert_topology_networkx(real_topology)
                heuristic_constraints = apply_heuristic(G, extra_channels=model_channel_counts)
                apply_opcode_heuristic(G, model_opcodes)

                solutions_iter = enumerate_solutions(G, None, fresh_context=True)
                next(solutions_iter)
                good = True

                break
            except Exception as e:
                # print(e)
                print(e, traceback.format_exc())
    
        if not good: return []
    # --------------------- Topological Perturbation -----------------------------------------

    else: 
        G = Graph.from_onnx(model)
        original_dot = G.make_dot()
        n_nodes = len(G.nodes)
        mutants = generate_topological_mutants(G, depth=max(2, n_nodes//2), count=5)
        G = random.choice(mutants)
        G.reset_edge_indices()
        heuristic_constraints = apply_heuristic(G, extra_channels=model_channel_counts)
        apply_opcode_heuristic(G, model_opcodes)

    # ----------------------------------------------------------------------------------------

    num_placeholders = G.num_placeholders()
    print("num_placeholder", num_placeholders)

    random.seed(time.time())
    time_begin = time.time()

    solutions_per_round = 100 if random_enumeration else max_solutions
    n_rounds = max_solutions // solutions_per_round

    manager = SolutionManager(G)
    solutions = []
    best_logprob_timeseries: List[float] = []
    logprob_timeseries = []

    try:
        for i in range(n_rounds):
            n_nodes = min(3, num_placeholders)
            extra_cons = get_random_initial_constraints(G, n_nodes=n_nodes, n_opcodes=15) if random_enumeration else []
            extra_cons.extend(initial_constraints)
            extra_cons.extend(heuristic_constraints)
            solutions_iter = enumerate_solutions(G, extra_cons, fresh_context=True)

            for idx in range(solutions_per_round):
                if not random_enumeration and idx % 1000 == 0:
                    print(
                        f"thread {thread_name}, round: {i + 1}, solutions: {len(manager.solutions)},"
                        f" rate: {len(manager.solutions) / (time.time() - time_begin):.2f} solns/s")
                try:
                    model, assignment, negation = next(solutions_iter)
                    # solutions.append(SerializableModel(model))
                    solutions.append(model)
                    logprob = manager.add_solution(model)
                    best_logprob_timeseries.append(manager.max_logprob)
                    logprob_timeseries.append(logprob)
                except StopIteration as e:
                    break
                except TimeoutError as e:
                    raise e
                except Exception as e:
                    print(e, traceback.format_exc(), file=sys.stderr, flush=True)
                    break

            if i % 10 == 0:
                print(
                    f"thread {thread_name}, round: {i + 1}, solutions: {len(manager.solutions)},"
                    f" rate: {len(manager.solutions) / (time.time() - time_begin):.2f} solns/s")

    except TimeoutError:
        print(f"thread {thread_name} timed out.")

    print(f"thread {thread_name} calculating statistics.")
    logprobs = manager.logprobs()

    time_end = time.time()

    if len(logprobs) > 0:
        os.makedirs(os.path.join(outdir, thread_name))

        plt.figure()
        plt.hist(logprobs, bins=50)
        plt.title(f"thread {thread_name} logprobs")
        plt.xlabel("logprob")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, thread_name, f"t{thread_name}_logprobs.pdf"))

        plt.figure()
        plt.plot(best_logprob_timeseries, c="red", label="best")
        plt.scatter(range(len(logprob_timeseries)), logprob_timeseries, s=2, label="logprobs")
        plt.legend()
        plt.xlabel("solution idx")
        plt.ylabel("logprob")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, thread_name, f"t{thread_name}_timeseries.pdf"))

        if original_dot:
            original_dot.render(os.path.join(outdir, thread_name, f"t{thread_name}_original"), cleanup=True)

        for rank in range(min(20, len(logprob_timeseries))):
            try:
                model, logprob = manager.get_rank(rank)
                dot = G.make_dot(model=model)
                dot.render(os.path.join(outdir, thread_name, f"t{thread_name}_graph_{rank}"), cleanup=True)
            except:
                print(f"thread {thread_name} rank {rank} failed to draw.")
                
        serialized_solutions = []
        for model in set(solutions):
            graph = G.make_networkx(model=model)
            graph_logprob = manager.calculate_logprob(model)
            graph.graph.update({
                "logprob": graph_logprob,
                "source": dict(real_topology.graph)
                })
            serialized_solutions.append(graph)

        with open(os.path.join(outdir, thread_name, "solutions.pkl"), "wb") as fp:
            pickle.dump(serialized_solutions, fp)

        with open(os.path.join(outdir, thread_name, "time.log"), "w") as fp:
            json.dump({
                "time_begin": time_begin,
                "time_end": time_end,
                "elapsed": time_end - time_begin
            }, fp)

    else:
        print(f"thread {thread_name} has nothing to do")

    print(f"thread {thread_name} done")


def worker_thread(args):
    thread_name, real_topology, serialized, max_solutions = args
    return thread_find_mutants_graphrnn(thread_name, real_topology, serialized, max_solutions)

if __name__ == '__main__':
    with open(args.real_pkl, "rb") as fp:
        reals = pickle.load(fp)
        reals = list(itertools.chain(*reals))
        reals = list(filter(lambda pair: len(pair[0].nodes) > 0, reals))

    max_solutions = 5000
    targs = []

    if not generate_new_topology:
        for idx, (g, s) in enumerate(reals):
            for m in range(num_mutants):
                targs.append((f"r{idx}m{m}", g, s, max_solutions))
    else:
        for idx, (g, s) in enumerate(reals):
            targs.append((f"r{idx}", g, s, max_solutions))

    with Pool(args.nthreads, maxtasksperchild=1) as pool:
        solutions = list(tqdm(pool.imap_unordered(worker_thread, targs), total=len(targs)))
