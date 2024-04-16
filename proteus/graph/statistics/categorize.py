from typing import List

categories = {
    "Abs": "EwUnary",
    "Mod": "EwBinary",
    "ConstantOfShape": "Constant",
    "AveragePool": "Pooling",
    "Ceil": "EwUnary",
    "NonZero": "EwUnary",
    "Relu": "Activation",
    "Erf": "EwUnary",
    "Exp": "EwUnary",
    "GatherND": "Gather",
    "Tanh": "Activation",
    "Resize": "Layout",
    "ScatterElements": "Layout",
    "ScatterND": "Layout",
    "Not": "EwUnary",
    "Softmax": "Normalization",
    "GlobalAveragePool": "Pooling",
    "BatchNormalization": "Normalization",
    "Sub": "EwBinary",
    "ReduceMean": "Reduction",
    "RoiAlign": "RoiAlign",
    "Xor": "EwUnary",
    "Slice": "Layout",
    "HardSwish": "Activation",
    "MaxPool": "Pooling",
    "Add": "EwBinary",
    "Reciprocal": "EwUnary",
    "Concat": "Layout",
    "Equal": "EwBinary",
    "Gemm": "MatMul",
    "Div": "EwBinary",
    "Einsum": "MatMul",
    "Constant": "Constant",
    "Less": "EwBinary",
    "Unsqueeze": "Layout",
    "Mul": "EwBinary",
    "Greater": "EwBinary",
    "Pad": "Layout",
    "ReduceProd": "Reduction",
    "Identity": "EwUnary",
    "ReduceMin": "Reduction",
    "ReduceSum": "Reduction",
    "And": "EwBinary",
    "Reshape": "Layout",
    "Log": "EwUnary",
    "HardSigmoid": "Activation",
    "Shape": "Layout",
    "Transpose": "Layout",
    "Cast": "Layout",
    "Squeeze": "Layout",
    "Tile": "Layout",
    "TopK": "Reduction",
    "Gather": "Layout",
    "Clip": "EwUnary",
    "Expand": "Layout",
    "If": "ControlFlow",
    "Loop": "ControlFlow",
    "Sigmoid": "Activation",
    "Min": "EwBinary",
    "ConvTranspose": "Convolution",
    "Conv": "Convolution",
    "Sqrt": "EwUnary",
    "GreaterOrEqual": "EwBinary",
    "InstanceNormalization": "Normalization",
    "Range": "ControlFlow",
    "MatMul": "MatMul",
    "Dropout": "Dropout",
    "Pow": "EwBinary",
    "Where": "EwBinary",
    "Floor": "EwUnary",
    "Split": "Layout",
    "Max": "EwBinary",
    "Flatten": "Layout",
    "ReduceMax": "Reduction",
    "ReduceL2": "Reduction",
    "LeakyRelu": "Activation"
}


def categorize_opcode(opcode: str):
    if opcode in categories:
        return categories[opcode]
    return opcode


def categorize_list(opcodes: List[str]):
    return list(map(categorize_opcode, opcodes))