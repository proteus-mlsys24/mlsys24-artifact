import random
import json
from typing import List, Dict, Union

import z3
from copy import deepcopy

from proteus.graph.graph import Graph, Placeholder
from proteus.graph.ops import OperatorRegistry


class SerializableModelEntry:
    def __init__(self, val: int):
        self.val = val

    def as_long(self):
        return self.val


class SerializableModel:
    def __init__(self, model: z3.ModelRef):
        self.model: Dict[str, SerializableModelEntry] = dict()
        for decl in model.decls():
            if isinstance(model[decl], z3.z3.IntNumRef):
                self.model[decl.name()] = SerializableModelEntry(model[decl].as_long())

    def __hash__(self):
        return hash(json.dumps(self.model, sort_keys=True))
    
    def __eq__(self, other):
        this = json.dumps(self.model, sort_keys=True)
        that = json.dumps(other.model, sort_keys=True)
        return this == that

    def __getitem__(self, item):
        if isinstance(item, z3.z3.FuncDeclRef):
            decl_name = item.name()
            return self.model[decl_name]

        if isinstance(item, z3.z3.ArithRef):
            decl_name = item.decl().name()
            return self.model[decl_name]

        if isinstance(item, str):
            return self.model[item]


ModelLike = Union[z3.ModelRef, SerializableModel]


def interpret_solution(G: Graph, model: z3.z3.ModelRef):
    placeholders: List[Placeholder] = []
    for node in G.nodes:
        if isinstance(node, Placeholder):
            placeholders.append(node)

    opcode_assignment: Dict[Placeholder, str] = dict()
    negated_assignment = list()
    for node in placeholders:
        opcode = model[node.node_opcode].as_long()
        opcode_name = OperatorRegistry.operators[opcode]
        opcode_assignment[node] = opcode_name

        negated_assignment.append(node.node_opcode == opcode)

    return opcode_assignment, z3.Not(z3.And(*negated_assignment))


def enumerate_solutions(G: Graph, extra_constraints=None, fresh_context=False):
    cons_lst = G.constraints()
    if extra_constraints is not None:
        cons_lst.extend(extra_constraints)

    cons = z3.And(*cons_lst)
    cons = z3.simplify(cons)

    if fresh_context:
        context = z3.Context()
        solver = z3.Solver()
        solver.add(cons)
        solver.translate(context)
    else:
        solver = z3.Solver()
        solver.add(cons)

    n_solutions = 0

    while solver.check() == z3.sat:
        model = solver.model()
        opcode_assignment, negation = interpret_solution(G, model)
        solver.add(negation)
        n_solutions += 1

        yield model, opcode_assignment, negation

    # print(f"Generated {n_solutions} solutions.")


def get_random_initial_constraints(G: Graph, n_nodes: int, n_opcodes: int):
    placeholders: List[Placeholder] = list(filter(lambda node: isinstance(node, Placeholder), G.nodes))
    placeholders = random.sample(placeholders, n_nodes)

    constraints = []
    for node in placeholders:
        candidates = random.sample(list(node.opcode_candidates), min(n_opcodes, len(node.opcode_candidates)))
        candidate_ids = list(map(lambda opcode: OperatorRegistry.operator_ids[opcode], candidates))

        constraints.append(z3.Or(*[node.node_opcode == c for c in candidate_ids]))

    return constraints
