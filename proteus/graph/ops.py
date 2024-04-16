
import z3

from typing import Type, Dict, List
from .graph import Placeholder
from .utils import try_get_long

"""
Still missing:
    Layout operators
        Split
        Concat
        Reshape
        Flatten
        Expand
        Gather
        Squeeze
        Transpose

    (done) Reduction operators

    (done) Activation
    
    (done) Normalization

    Convolution/Pooling
        (done) ConvTranspose

    Special operators
        (done) Constant
"""

class OpBase:
    def __init__(self, placeholder: Placeholder):
        self.placeholder = placeholder

    def constraints(self):
        return list()

    def label(self, model):
        return self.__class__.__name__

    def get_attr_dict(self, model):
        return {}


class OperatorRegistry:
    operators: List[str] = list()
    operator_lookup: Dict[str, Type[OpBase]] = dict()
    operator_ids: Dict[str, int] = dict()

def register_op(op_cls):
    op_id = len(OperatorRegistry.operator_ids)
    name = op_cls.__name__
    OperatorRegistry.operators.append(name)
    OperatorRegistry.operator_lookup[name] = op_cls
    OperatorRegistry.operator_ids[name] = op_id
    return op_cls

class SingleOutputOp(OpBase):
    def constraints(self):
        cons = super().constraints()

        # the shape of all the output match
        for output_id in range(1, self.placeholder.num_outputs):
            for dim_id in range(self.placeholder.max_dims):
                cons.append(
                    self.placeholder.output_dims[0][dim_id] == self.placeholder.output_dims[output_id][dim_id]
                )

        return cons

@register_op
class Constant(SingleOutputOp):
    num_inputs = 0

@register_op
class ConstantOfShape(SingleOutputOp):
    num_inputs = 0

class Flatten(SingleOutputOp):
    num_inputs = 1

    def constraints(self):
        cons = super().constraints()

        # number of items should be consistent
        input_num_elems = 1
        output_num_elems = 1
        for dim_id in range(1, self.placeholder.max_dims):
            input_num_elems *= self.placeholder.input_dims[0][dim_id]
            output_num_elems *= self.placeholder.output_dims[0][dim_id]

        cons.append(input_num_elems == output_num_elems)

        return cons

@register_op
class MatMul(SingleOutputOp):
    num_inputs = 2

    def constraints(self):
        cons = super().constraints()

        # leading dimensions of input and outputs must match
        for dim_id in range(2, self.placeholder.max_dims):
            cons.append(self.placeholder.input_dims[0][dim_id] == self.placeholder.input_dims[1][dim_id])
            cons.append(self.placeholder.input_dims[0][dim_id] == self.placeholder.output_dims[0][dim_id])

        # reduction dimension must match
        cons.append(self.placeholder.input_dims[0][-1] == self.placeholder.input_dims[1][-2])

        # output dimension matches the outer dimensions of the input
        cons.append(self.placeholder.output_dims[0][-2] == self.placeholder.input_dims[0][-2])
        cons.append(self.placeholder.output_dims[0][-1] == self.placeholder.input_dims[1][-1])

        return cons
    
@register_op
class Conv(SingleOutputOp):
    num_inputs = 1

    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.in_channels = z3.Int(f"{self.placeholder.node_name}_in_channels")
        self.out_channels = z3.Int(f"{self.placeholder.node_name}_out_channels")
        self.kernel_size = z3.Int(f"{self.placeholder.node_name}_kernel_size")
        self.padding = z3.Int(f"{self.placeholder.node_name}_padding")
        self.stride = z3.Int(f"{self.placeholder.node_name}_stride")

    def constraints(self):
        cons = super().constraints()

        """
        image input shape is:

            -3       -2     -1
        (* channels height width)

        """

        # kernel size greater than 0
        cons.append(self.kernel_size > 0)
        cons.append(self.stride > 0)
        cons.append(self.padding >= 0)
        cons.append(self.in_channels > 0)
        cons.append(self.out_channels > 0)

        # TODO: Remove this constraint. Conv must have common kernel sizes
        cons.append(z3.Or(*[ self.kernel_size == k for k in [1, 3, 5, 7, 9, 11]]))

        # TODO: Remove this constraint. Conv must have common paddings and strides
        cons.append(z3.Or(self.stride == 1, self.stride == 2))
        cons.append(z3.Or(self.padding == 0, 2 * self.padding + 1 == self.kernel_size))

        # check channel count
        if len(self.placeholder.input_dims) > 0:
            cons.append(self.placeholder.input_dims[0][-3] == self.in_channels)
        if len(self.placeholder.output_dims) > 0:
            cons.append(self.placeholder.output_dims[0][-3] == self.out_channels)

        # check height and width
        if len(self.placeholder.input_dims) > 0 and len(self.placeholder.output_dims) > 0:
            for dim_id in [-2, -1]:
                # cons.append(self.placeholder.input_dims[0][dim_id] - self.kernel_size == self.placeholder.output_dims[0][dim_id])

                cons.append(
                    (self.placeholder.input_dims[0][dim_id] + 2 * self.padding - self.kernel_size) / self.stride + 1 ==
                    self.placeholder.output_dims[0][dim_id]
                )

        return cons

    def label(self, model):
        in_channels = model[self.in_channels].as_long()
        out_channels = model[self.out_channels].as_long()
        kernel_size = model[self.kernel_size].as_long()
        padding = model[self.padding].as_long()
        stride = model[self.stride].as_long()

        return f"{self.__class__.__name__} cin: {in_channels}, cout: {out_channels}, ker: {kernel_size}, pad: {padding}, stride: {stride}"

    def get_attr_dict(self, model):
        return {
            "in_channels": try_get_long(model[self.in_channels]),
            "out_channels": try_get_long(model[self.out_channels]),
            "kernel_size": try_get_long(model[self.kernel_size]),
            "padding": try_get_long(model[self.padding]),
            "stride": try_get_long(model[self.stride]),
        }


@register_op
class ConvTranspose(SingleOutputOp):
    num_inputs = 1

    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.in_channels = z3.Int(f"{self.placeholder.node_name}_in_channels")
        self.out_channels = z3.Int(f"{self.placeholder.node_name}_out_channels")
        self.kernel_size = z3.Int(f"{self.placeholder.node_name}_kernel_size")
        # self.stride = z3.Int(f"{self.placeholder.node_name}_stride")

    def constraints(self):
        cons = super().constraints()

        """
        image input shape is:

            -3       -2     -1
        (* channels height width)

        """

        # kernel size greater than 0
        cons.append(self.kernel_size > 0)

        # TODO: Remove this constraint. Conv must have common kernel sizes
        cons.append(z3.Or(*[ self.kernel_size == k for k in [1, 3, 5, 7, 9, 11]]))

        # check channel count
        if len(self.placeholder.input_dims) > 0:
            cons.append(self.placeholder.input_dims[0][-3] == self.in_channels)
        if len(self.placeholder.output_dims) > 0:
            cons.append(self.placeholder.output_dims[0][-3] == self.out_channels)

        # check height and width
        if len(self.placeholder.input_dims) > 0 and len(self.placeholder.output_dims) > 0:
            for dim_id in [-2, -1]:
                cons.append(self.placeholder.input_dims[0][dim_id] + self.kernel_size - 1 == self.placeholder.output_dims[0][dim_id])

        return cons

    def label(self, model):
        in_channels = model[self.in_channels].as_long()
        out_channels = model[self.out_channels].as_long()
        kernel_size = model[self.kernel_size].as_long()

        return f"{self.__class__.__name__} cin: {in_channels}, cout: {out_channels}, ker: {kernel_size}"

    def get_attr_dict(self, model):
        return {
            "in_channels": try_get_long(model[self.in_channels]),
            "out_channels": try_get_long(model[self.out_channels]),
            "kernel_size": try_get_long(model[self.kernel_size]),
        }

class PoolingOp(SingleOutputOp):
    num_inputs = 1

    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.kernel_size = z3.Int(f"{self.placeholder.node_name}_kernel_size")
        self.stride = z3.Int(f"{self.placeholder.node_name}_stride")
        self.padding = z3.Int(f"{self.placeholder.node_name}_padding")

    def constraints(self):
        cons = super().constraints()

        """
        image input shape is:

            -3       -2     -1
        (* channels height width)

        """

        # kernel size greater than 0
        cons.append(self.kernel_size > 0)
        cons.append(self.stride > 0)
        cons.append(self.padding >= 0)

        # TODO: Remove this constraint. Pooling must have common kernel sizes
        cons.append(z3.Or(*[ self.kernel_size == k for k in [1, 3, 5, 7, 9, 11]]))

        # TODO: Remove this constraint. Pooling must have common paddings and strides
        cons.append(z3.Or(self.stride == 1, self.stride == 2))
        cons.append(z3.Or(self.padding == 0, 2 * self.padding + 1 == self.kernel_size))

        # check height and width
        if len(self.placeholder.input_dims) > 0 and len(self.placeholder.output_dims) > 0:
            for dim_id in [-2, -1]:
                # cons.append(self.placeholder.input_dims[0][dim_id] - self.kernel_size == self.placeholder.output_dims[0][dim_id])

                cons.append(
                    (self.placeholder.input_dims[0][dim_id] + 2 * self.padding - self.kernel_size) / self.stride + 1 ==
                    self.placeholder.output_dims[0][dim_id]
                )

        return cons

    def label(self, model):
        kernel_size = model[self.kernel_size].as_long()
        padding = model[self.padding].as_long()
        stride = model[self.stride].as_long()

        return f"{self.__class__.__name__} ker: {kernel_size}, pad: {padding}, stride: {stride}"

    def get_attr_dict(self, model):
        return {
            "kernel_size": try_get_long(model[self.kernel_size]),
            "padding": try_get_long(model[self.padding]),
            "stride": try_get_long(model[self.stride]),
        }



class GlobalPoolingOp(SingleOutputOp):
    num_inputs = 1

    def __init__(self, placeholder):
        super().__init__(placeholder)

    def constraints(self):
        cons = super().constraints()

        """
        why not inherit PoolingOp? 
        global pooling effectively is pooling with kernel size = spatial dimensions
        but spatial dimensions are often non-square but with our current config it needs to
        be square. it just adds more pain than convenience.
        
        instead we expect the spatial dimensions to be 1
        """

        # check height and width
        if len(self.placeholder.input_dims) > 0 and len(self.placeholder.output_dims) > 0:
            for dim_id in [-2, -1]:
                cons.append(1 == self.placeholder.output_dims[0][dim_id])

        return cons


class EWUnaryOp(SingleOutputOp):
    num_inputs = 1

    def constraints(self):
        cons = super().constraints()
        # inputs and outputs have the same shapes
        if len(self.placeholder.input_dims) > 0 and len(self.placeholder.output_dims) > 0:
            for dim_id in range(self.placeholder.max_dims):
                cons.append(self.placeholder.input_dims[0][dim_id] == self.placeholder.output_dims[0][dim_id])

        return cons

class EWBinaryOp(SingleOutputOp):
    num_inputs = 2

    def constraints(self):
        cons = super().constraints()
        # inputs have the same shapes
        if len(self.placeholder.input_dims) > 0 and len(self.placeholder.output_dims) > 0:
            for dim_id in range(self.placeholder.max_dims):
                cons.append(self.placeholder.input_dims[0][dim_id] == self.placeholder.input_dims[1][dim_id])

        # inputs and outputs have the same shapes
        if len(self.placeholder.input_dims) > 0 and len(self.placeholder.output_dims) > 0:
            for dim_id in range(self.placeholder.max_dims):
                cons.append(self.placeholder.input_dims[0][dim_id] == self.placeholder.output_dims[0][dim_id])

        return cons

class ReductionOp(SingleOutputOp):
    num_inputs = 1

    def __init__(self, placeholder):
        super().__init__(placeholder)
        self.reduction_dim = z3.Int(f"{self.placeholder.node_name}_reduction_dim")
        self.keepdims = z3.Bool(f"{self.placeholder.node_name}_keepdims")

    def constraints(self):
        cons = super().constraints()

        # reduction dim must be one of the dimensions
        cons.append(z3.Or(*[self.reduction_dim == dim for dim in range(Placeholder.max_dims)]))

        # the correct dimension has been reduced
        for reduction_dim in range(self.placeholder.max_dims):
            for keepdims in [False, True]:
                expected_output = list(self.placeholder.input_dims[0])

                if keepdims:
                    expected_output[reduction_dim] = 1
                else:
                    expected_output.pop(reduction_dim)
                    expected_output.append(0)

                if len(self.placeholder.output_dims) > 0:
                    cons.append(z3.Implies(
                        self.reduction_dim == reduction_dim and self.keepdims == keepdims,
                        z3.And(*(
                            self.placeholder.output_dims[0][i] == expected_output[i]
                            for i in range(self.placeholder.max_dims)
                        ))
                    ))

        return cons

    def label(self, model):
        reduction_dim = model[self.reduction_dim].as_long()
        # keepdims = model[self.keepdims].as_long()

        # return f"{self.__class__.__name__} dim: {reduction_dim}, keep: {keepdims}"
        return f"{self.__class__.__name__} dim: {reduction_dim}"

    def get_attr_dict(self, model):
        return {
            "reduction_dim": try_get_long(model[self.reduction_dim]),
        }

# Elementwise Binary Operations
@register_op
class Add(EWBinaryOp): pass

@register_op
class Sub(EWBinaryOp): pass

@register_op
class Mul(EWBinaryOp): pass

@register_op
class Div(EWBinaryOp): pass

@register_op
class Pow(EWBinaryOp): pass

@register_op
class Max(EWBinaryOp): pass

@register_op
class Min(EWBinaryOp): pass

# Elementwise Unary Operations
@register_op
class Abs(EWUnaryOp): pass

@register_op
class Neg(EWUnaryOp): pass

@register_op
class Ceil(EWUnaryOp): pass

@register_op
class Floor(EWUnaryOp): pass

@register_op
class Round(EWUnaryOp): pass

@register_op
class Sqrt(EWUnaryOp): pass

@register_op
class Exp(EWUnaryOp): pass

@register_op
class Log(EWUnaryOp): pass

@register_op
class Sin(EWUnaryOp): pass

@register_op
class Cos(EWUnaryOp): pass

@register_op
class Tan(EWUnaryOp): pass

@register_op
class Asin(EWUnaryOp): pass

@register_op
class Acos(EWUnaryOp): pass

@register_op
class Atan(EWUnaryOp): pass

@register_op
class Reciprocal(EWUnaryOp): pass

@register_op
class Identity(EWUnaryOp): pass

# Activation Functions
@register_op
class Sigmoid(EWUnaryOp): pass

@register_op
class Relu(EWUnaryOp): pass

@register_op
class LeakyRelu(EWUnaryOp): pass

@register_op
class Selu(EWUnaryOp): pass

@register_op
class Elu(EWUnaryOp): pass

@register_op
class Tanh(EWUnaryOp): pass

@register_op
class HardSigmoid(EWUnaryOp): pass

@register_op
class Softmax(EWUnaryOp): pass

@register_op
class LogSoftmax(EWUnaryOp): pass

# Normalization Operators
@register_op
class BatchNormalization(EWUnaryOp): pass

@register_op
class InstanceNormalization(EWUnaryOp): pass

@register_op
class LayerNormalization(EWUnaryOp): pass

@register_op
class LRN(EWUnaryOp): pass

@register_op
class GroupNormalization(EWUnaryOp): pass

@register_op
class InstanceNormalization(EWUnaryOp): pass

# Pooling Operators
@register_op
class AveragePool(PoolingOp): pass

@register_op
class MaxPool(PoolingOp): pass

@register_op
class GlobalAveragePool(GlobalPoolingOp): pass

@register_op
class GlobalMaxPool(GlobalPoolingOp): pass

@register_op
class LpPool(PoolingOp): pass

@register_op
class AdaptiveAveragePool(PoolingOp): pass

@register_op
class AdaptiveMaxPool(PoolingOp): pass

# Reduction Operators
@register_op
class ReduceSum(ReductionOp): pass

@register_op
class ReduceMean(ReductionOp): pass

@register_op
class ReduceMax(ReductionOp): pass

@register_op
class ReduceMin(ReductionOp): pass

@register_op
class ReduceProd(ReductionOp): pass
