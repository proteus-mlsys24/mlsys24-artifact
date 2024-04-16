
import argparse
import torch
import itertools
import pickle
import random
import os
from typing import List, Set, Tuple
from dataset import ModelDataset
from gcn import GCN
import pytorch_lightning as pl
import networkx as nx
from functools import partial

from train import load_consolidated_fakes, load_reals, chain, collect_opcodes, scramble_opcodes
from gcn import GCN

def filter_data(graph: nx.DiGraph, model_name: str, negate=False):
    """
    {'logprob': -3.2355976960410917,
    'source': {'onnx_file': 'onnx_cache/densenet121_1_3_224_224.onnx',
    'partition_id': 36}}
    """
    if "source" in graph.graph:
        full_filename = graph.graph["source"]["onnx_file"]
        model_filename = os.path.basename(full_filename)
    else:
        full_filename = graph.graph["onnx_file"]
        model_filename = os.path.basename(full_filename)

    # this is a special case for language models where the path looks like model_name/model.onnx
    if model_filename == "model.onnx":
        model_filename = full_filename.split("/")[-2]

    return negate == (model_filename.startswith(model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--randomize_opcode", action='store_true')
    parser.add_argument("--figure_dir", type=str)
    args = parser.parse_args()
    
    graphs, labels = chain([
        load_consolidated_fakes("consolidated_fakes.pkl"),
        load_reals("real.pkl")
    ])
    opcodes = list(collect_opcodes(graphs))
    if args.randomize_opcode: scramble_opcodes(graphs, labels, opcodes)
 
    dataset = ModelDataset(graphs, labels)
    model = GCN(
        input_dims=len(dataset.opcodes),
        hidden_dims=128,
        figure_dir=args.figure_dir
    )
    model.opcode_mapping.update(dataset.opcode_indices)
 
    train = ModelDataset(graphs, labels, 
                         filter_fn=partial(filter_data, model_name=args.model_name, negate=False),
                         opcode_indices=dataset.opcode_indices)
    
    val = ModelDataset(graphs, labels, 
                         filter_fn=partial(filter_data, model_name=args.model_name, negate=True),
                         opcode_indices=dataset.opcode_indices)
    
    trainer = pl.Trainer(
        accumulate_grad_batches=8,
        max_epochs=5
        # gradient_clip_val=1e-2
    )
    trainer.fit(model, train, val)
