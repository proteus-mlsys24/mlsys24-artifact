import random
import itertools
import os
import time
import multiprocessing
import pickle
from copy import deepcopy
from proteus.graph import Graph
from proteus.graph.solver import enumerate_solutions, get_random_initial_constraints, SerializableModel
from proteus.graph.statistics.manager import SolutionManager

from tqdm import tqdm
import matplotlib.pyplot as plt

from pathos.multiprocessing import ProcessingPool


def thread_make_solutions(G: Graph, max_solutions: int, thread_id: int):
    random.seed(time.time())
    G_clone = deepcopy(G)
    extra_cons = get_random_initial_constraints(G_clone, n_nodes=3, n_opcodes=15)
    solutions_iter = enumerate_solutions(G_clone, extra_cons, fresh_context=True)

    solutions = []
    for idx in range(max_solutions):
        try:
            model, assignment = next(solutions_iter)
            solutions.append(SerializableModel(model))
        except StopIteration:
            break

    return solutions


def parallel_solve(G: Graph, n_threads: int, n_tasks: int, n_solutions_per_task: int, outdir: str = None):
    def thread_make_solutions_2(thread_id: int):
        random.seed(time.time())
        G_clone = deepcopy(G)
        extra_cons = get_random_initial_constraints(G_clone, n_nodes=3, n_opcodes=15)
        solutions_iter = enumerate_solutions(G_clone, extra_cons, fresh_context=True)

        solutions = []
        for idx in range(n_solutions_per_task):
            try:
                model, assignment = next(solutions_iter)
                solutions.append(SerializableModel(model))
            except StopIteration:
                break

        return solutions


    time_begin = time.time()
    manager = SolutionManager(G)
    # with multiprocessing.Pool(n_threads) as pool:
    with ProcessingPool(nodes=n_threads) as pool:
        # thread_args = itertools.product(
        #     [G],
        #     [n_solutions_per_task],
        #     range(n_tasks)
        # )
        # task_iter = pool.imap(thread_make_solutions, thread_args)
        task_iter = pool.imap(thread_make_solutions_2, range(n_tasks))

        bar = tqdm(task_iter, total=n_tasks, desc="Generating Solutions")
        for subset in bar:
            for solution in subset:
                manager.add_solution(solution)
            bar.write(
                f"New: {len(subset)}. Total: {len(manager.solutions)}. Avg: {len(manager.solutions) / (time.time() - time_begin):.3f} solns/second.")

    logprobs = manager.logprobs()

    print("logprobs", logprobs)

    if outdir is not None:
        os.makedirs(outdir)

        plt.figure()
        plt.hist(logprobs, bins=50)
        plt.xlabel("logprob")
        plt.savefig(os.path.join(outdir, "logprobs.pdf"))

        for rank in list(range(20)) + list(range(1000, len(logprobs), 1000)):
            model, logprob = manager.get_rank(rank)
            print("rank", rank, "logprob", logprob)
            dot = G.make_dot(model=model)
            dot.render(os.path.join(outdir, f"graph_{rank}"), cleanup=True)
