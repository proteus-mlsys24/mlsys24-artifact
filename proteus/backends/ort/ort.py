
import onnxruntime as ort
import torch
import os
import sys
import onnx
from tempfile import mkdtemp

optimization_levels = {
    "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL
}

def printc(text):
    from termcolor import colored
    print(colored(text, "yellow"), flush=True, file=sys.stderr)

# ------------------
# LM input vars
batch_size = 2
sequence_length = 256
lm_inputs = {
    # "input_ids": torch.zeros(batch_size, sequence_length).long(),
    # "token_type_ids": torch.zeros(batch_size, sequence_length).long(),
    # "attention_mask": torch.ones(batch_size, sequence_length).long(),
}
# ------------------

def optimize_graph(model):
    sess_options = ort.SessionOptions()
    outdir = mkdtemp()

    optimized_models = dict()

    for name, opt_level in optimization_levels.items():
        sess_options.graph_optimization_level = opt_level
        sess_options.optimized_model_filepath = os.path.join(outdir, f"opt_{name}.onnx")
        sess = ort.InferenceSession(model.SerializeToString(), sess_options, providers=["CUDAExecutionProvider"])
        
        optimized = onnx.load(sess_options.optimized_model_filepath)
        optimized_models[name] = optimized

    return optimized_models

def optimize_partitions(ps, opt_level="all"):
    for idx, p in enumerate(ps.partitions):
        model = ps.partitions[idx].model
        try:
            optimized = optimize_graph(model)[opt_level]
            ps.mutate(idx, optimized)
        except Exception as e:
            print("Optimization failed:", e)

def profile_one(filename, optim_level, output_prefix, reps=100):
    try:
        onnx_model = onnx.load(filename)
        print(onnx_model.graph.input)
        # onnx.checker.check_model(onnx_model)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = optimization_levels[optim_level]
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = output_prefix

        printc("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        sess = ort.InferenceSession(filename, 
                                    providers=["CUDAExecutionProvider"], 
                                    sess_options=sess_options)
        printc("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
        inputs = {}
        for i in onnx_model.graph.input:
            input_name = i.name

            if input_name not in lm_inputs:
                input_shape = [ d.dim_value for d in i.type.tensor_type.shape.dim ]
                if i.type.tensor_type == 7:
                    # handle int64 cases
                    inputs[input_name] = torch.rand(*input_shape).long().numpy()
                else:
                    inputs[input_name] = torch.rand(*input_shape).numpy()
                
            else:
                inputs[input_name] = lm_inputs[input_name].numpy()

        printc("--------------------------")
        for k in inputs:
            printc(f"{k}, {inputs[input_name].shape}")
        printc("--------------------------")

        for _ in range(reps):
            outputs = sess.run(None, inputs)

        sess.end_profiling()

        return True
    except Exception as e:
        print("Failed to profile:", e)
        print("Filename that failed:", filename)

        return False