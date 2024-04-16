from typing import List, Dict, Union, Optional, Set
import z3

from .onnx.utils import get_tensor_shapes, extract_attributes

import sys
import onnx
import graphviz
from termcolor import colored

OPCODE_WHITELIST = {'Conv', 'Relu', 'MaxPool', 'Add', 'GlobalAveragePool', 'Flatten', 'Gemm', 'Constant', 'Pad',
                    'AveragePool', 'Concat', 'Shape', 'Slice', 'Resize', 'HardSwish', 'HardSigmoid', 'Mul', 'Reshape',
                    'Transpose', 'Gather', 'Div', 'ReduceMean', 'Split', 'Squeeze', 'Sub', 'ReduceMin', 'Cast',
                    'ReduceMax', 'Reciprocal', 'Min', 'Unsqueeze', 'Floor', 'Ceil', 'ConstantOfShape', 'Range',
                    'Expand', 'Clip', 'Exp', 'TopK', 'Sigmoid', 'Max', 'GreaterOrEqual', 'And', 'NonZero', 'ReduceProd',
                    'Equal', 'If', 'Sqrt', 'Log', 'RoiAlign', 'Where', 'ScatterElements', 'Softmax', 'Greater',
                    'ConvTranspose', 'Loop', 'Pow', 'MatMul', 'Erf', 'Mod', 'ScatterND', 'Abs', 'ReduceSum', 'Not',
                    'BatchNormalization', 'InstanceNormalization', 'GatherND', 'Less', 'Xor', 'Tile', 'Einsum', 'Tanh'}


class FixedNode:
    last_id = 0

    def __init__(self, onnx_node: onnx.NodeProto, input_shapes: List[List[int]], output_shape: List[int]):
        self.node_name = f"fixed_{FixedNode.last_id}"
        FixedNode.last_id += 1

        # self.onnx_node = onnx_node
        self.onnx_node_serialized = onnx_node.SerializeToString()
        self.input_shapes = input_shapes
        self.output_shape = output_shape

    @property
    def onnx_node(self):
        return onnx.NodeProto.FromString(self.onnx_node_serialized)

    def get_attr_dict(self):
        other_attrs = extract_attributes(self.onnx_node)
        ret = {
            "node_type": "FixedNode",
            "opcode": str(self.onnx_node.op_type),
            # "input_shapes": self.input_shapes,
            # "output_shape": self.output_shape,
            **other_attrs
        }

        return ret

    def label(self):
        return f"{self.onnx_node.op_type}\nin:{self.input_shapes}\nout:{self.output_shape}"

    def __str__(self):
        return f"{self.onnx_node.op_type}, in:{self.input_shapes}, out:{self.output_shape}"

    def __repr__(self):
        return self.__str__()


class Placeholder:
    last_id = 0
    min_dims = 1
    max_dims = 4

    def __init__(self, num_inputs: int, num_outputs: int, replacement_node: FixedNode = None):
        # Placed here to avoid circular import
        from .ops import OperatorRegistry, OpBase

        self.node_name = f"placeholder_{Placeholder.last_id}"
        Placeholder.last_id += 1

        self.node_opcode = z3.Int(f"{self.node_name}_opcode")

        self.num_inputs = num_inputs
        self.num_outputs = max(1, num_outputs)
        self.replacement_node = replacement_node
        self.whitelist: Optional[Set[str]] = None

        self.input_dims = [
            [
                z3.Int(f"{self.node_name}_input{input_id}_dim{dim_id}")
                for dim_id in range(Placeholder.max_dims)
            ]
            for input_id in range(self.num_inputs)
        ]

        self.output_dims = [
            [
                z3.Int(f"{self.node_name}_output{output_id}_dim{dim_id}")
                for dim_id in range(Placeholder.max_dims)
            ]
            for output_id in range(self.num_outputs)
        ]

        self.opcode_candidates: Dict[str, OpBase] = dict()

        for idx, opcode in enumerate(OperatorRegistry.operators):
            if opcode not in OPCODE_WHITELIST: continue

            opcode_cls = OperatorRegistry.operator_lookup[opcode]

            # filter
            if opcode_cls.num_inputs != self.num_inputs: continue
            # if opcode_cls.num_outputs != self.num_outputs: continue

            # add
            # print(f"{self.node_name} could be {opcode}")
            # if opcode == "Relu":
            self.opcode_candidates[opcode] = opcode_cls(self)

        # print("opcode candidates", list(self.opcode_candidates.keys()))

    def constraints(self):
        from proteus.graph.ops import OperatorRegistry
        cons = []

        valid_opcode_ids = []
        for opcode in self.opcode_candidates:
            if self.whitelist and opcode not in self.whitelist: continue
            valid_opcode_ids.append(OperatorRegistry.operator_ids[opcode])

        cons.append(z3.Or([self.node_opcode == opcode_id for opcode_id in valid_opcode_ids]))

        for opcode in self.opcode_candidates:
            if self.whitelist and opcode not in self.whitelist: continue
            opcode_id = OperatorRegistry.operator_ids[opcode]
            placeholder = self.opcode_candidates[opcode]

            cons.append(z3.Implies(
                self.node_opcode == opcode_id,
                z3.And(*placeholder.constraints())
            ))

        # print(f"Found {len(self.opcode_candidates)} candidates.")
        return cons

    def label(self, model=None):
        if model is None: return self.node_name
        from proteus.graph.ops import OperatorRegistry

        opcode_id = model[self.node_opcode].as_long()
        opcode = OperatorRegistry.operators[opcode_id]
        placeholder = self.opcode_candidates[opcode]
        placeholder_label = placeholder.label(model)

        return f"{self.node_name}\n{placeholder_label}"

    def get_attr_dict(self, model):
        from proteus.graph import OperatorRegistry
        opcode_id = model[self.node_opcode].as_long()
        opcode_name = OperatorRegistry.operators[opcode_id]

        opcode_attrs = {}
        if opcode_name in self.opcode_candidates:
            opcode_attrs = self.opcode_candidates[opcode_name].get_attr_dict(model)

        input_shapes = [
            [
                model[self.input_dims[input_id][dim_id]].as_long()
                if self.input_dims[input_id][dim_id] in model else None
                for dim_id in range(self.max_dims)
            ] for input_id in range(self.num_inputs)
        ]
        output_shapes = [
            [
                model[self.output_dims[output_id][dim_id]].as_long()
                if self.output_dims[output_id][dim_id] in model else None
                for dim_id in range(self.max_dims)
            ] for output_id in range(self.num_outputs)
        ]

        ret = {
            "node_type": "Placeholder",
            "opcode_id": opcode_id,
            "opcode": OperatorRegistry.operators[opcode_id],
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            **opcode_attrs
        }

        return ret



GraphNode = Union[FixedNode, Placeholder]
