import pickle
import z3
from typing import Dict, Tuple
import numpy as np

from proteus.graph import Graph, Placeholder, OperatorRegistry
from proteus.graph.solver import ModelLike, SerializableModel
from proteus.graph.statistics.pathprob import path_conditional_probability, get_paths
from proteus.graph.statistics.categorize import categorize_list
from proteus.config import ProteusConfig


class SolutionManager:
    def __init__(self, G: Graph):
        self.sequence_freqs: Dict[Tuple[str], int] = dict()
        with open(ProteusConfig.sequence_freqs_path, "rb") as fp:
            self.sequence_freqs = pickle.load(fp)

        self.category_freqs: Dict[Tuple[str], int] = dict()
        for p, f in self.sequence_freqs.items():
            p = tuple(categorize_list(list(p)))
            if p not in self.category_freqs: self.category_freqs[p] = 0
            self.category_freqs[p] += f
            
        with open(ProteusConfig.opcode_freqs_path, "rb") as fp:
            self.opcode_freqs: Dict[str, int] = pickle.load(fp)

        self.sorted = False
        self.solutions = list()
        self.G = G
        self.paths = get_paths(G, 3)
        self.max_logprob = None

    def calculate_logprob(self, model: ModelLike):
        # calculate opcode sequence prob
        logprobs = []
        for path in self.paths:
            numerator, denominator = path_conditional_probability(path, model, self.sequence_freqs)
            prob = numerator / denominator
            logprobs.append(np.log(prob))
        avg_logprob = np.average(logprobs)

        # calculate category sequence prob
        category_logprobs = []
        for path in self.paths:
            numerator, denominator = path_conditional_probability(path, model, self.category_freqs,
                                                                  process_fn=categorize_list)
            prob = numerator / denominator
            category_logprobs.append(np.log(prob))
        avg_category_logprob = np.average(category_logprobs)

        # calculate opcode probs
        opcode_logprobs = []
        opcode_denominator = sum(self.opcode_freqs.values())
        for node in self.G.nodes:
            if isinstance(node, Placeholder):
                opcode_id = model[node.node_opcode].as_long()
                opcode = OperatorRegistry.operators[opcode_id]
                opcode_freq = 1
                if opcode in self.opcode_freqs:
                    opcode_freq += self.opcode_freqs[opcode]

                opcode_logprobs.append(np.log(opcode_freq / opcode_denominator))

        avg_opcode_logprob = np.average(opcode_logprobs)

        # consider the product of the three (same thing as adding logprobs)
        logprob = avg_logprob + avg_category_logprob + avg_opcode_logprob

        return logprob

    def add_solution(self, model: ModelLike):
        logprob = self.calculate_logprob(model)
        if not np.isnan(logprob):
            self.sorted = False
            self.solutions.append((model, logprob))

            if self.max_logprob is None or self.max_logprob < logprob:
                self.max_logprob = logprob

        return logprob

    def deduplicate(self):
        deduped = []
        size_before = len(self.solutions)
        models_seen = set()
        for model, logprob in self.solutions:
            if model in models_seen: continue
            models_seen.add(model)
            deduped.append((model, logprob))

        self.solutions = deduped
        print(f"deduplicate. before: {size_before}, after: {len(deduped)}")


    def logprobs(self):
        return list(map(lambda t: t[1], self.solutions))

    def get_rank(self, rank: int):
        if not self.sorted:
            self.solutions.sort(key=lambda x: x[1], reverse=True)
            self.sorted = True

        return self.solutions[rank]
