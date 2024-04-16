
import onnx
import itertools

def extract_shape_from_type_proto(type_proto):
    if type_proto.HasField('tensor_type'):
        tensor_type = type_proto.tensor_type
        shape = [dim.dim_value if dim.dim_value > 0 else None for dim in tensor_type.shape.dim]
        return shape
    
def extract_shape_from_tensor_proto(tensor_proto):
    return tensor_proto.dims

def get_tensor_shapes(model: onnx.ModelProto):
    tensor_shapes = dict()

    for vi in itertools.chain(model.graph.input, model.graph.output, model.graph.value_info):
        tensor_shapes[vi.name] = extract_shape_from_type_proto(vi.type)

    for t in model.graph.initializer:
        tensor_shapes[t.name] = extract_shape_from_tensor_proto(t)

    return tensor_shapes

def extract_attributes(node: onnx.NodeProto):
    props = dict()
    for a in node.attribute:
        prop_name = a.name
        prop_value = None

        if a.type == onnx.AttributeProto.INTS:
            prop_value = list(a.ints)
        elif a.type == onnx.AttributeProto.INT:
            prop_value = int(a.i)

        props[prop_name] = prop_value


    return props

def topologically_sort_nodes(model: onnx.ModelProto):
    in_degree = dict()
    output_of = dict()
    edges = []
    next_nodes = dict()
    node_lookup: Dict[str, onnx.NodeProto] = dict()
    for node in model.graph.node:
        node_lookup[node.name] = node
        in_degree[node.name] = 0
        next_nodes[node.name] = []
        for o in node.output:
            output_of[o] = node.name

    global_inputs = list()
    for graph_input in itertools.chain(model.graph.input, model.graph.initializer):
        global_inputs.append(graph_input.name)

    for node in model.graph.node:
        for i in node.input:
            if i in output_of:
                src_node = output_of[i]
                edges.append((src_node, node.name))
                next_nodes[src_node].append(node.name)
                in_degree[node.name] += 1
            elif i in global_inputs:
                pass
            else:
                print(f"not found input {i}")

    remainder = set(in_degree.keys())
    ordering = list()

    while len(remainder) > 0:
        to_delete = []
        for r in remainder:
            if in_degree[r] == 0:
                to_delete.append(r)
                ordering.append(node_lookup[r])

                for next_node in next_nodes[r]:
                    in_degree[next_node] -= 1

        remainder -= set(to_delete)

        if len(to_delete) == 0:
            raise Exception("Graph contains cycles.")

    model.graph.node.clear()
    model.graph.node.extend(ordering)

def replace_mem_instrs_with_identity(model: onnx.ModelProto):
    memory_opcodes = [
        "MemcpyFromHost",
        "MemcpyToHost",
    ]
    for node in model.graph.node:
        if node.op_type in memory_opcodes:
            node.op_type = "Identity"