
import os
from glob import glob

class ProteusConfig:
    # Path to your GraphRNN installation.
    graphrnn_path = "/root/mlsys24-artifact/third_party/GraphRNN"

    # The set of GraphRNN generated graphs that we can use
    graphrnn_graphs = list(glob("/root/mlsys24-artifact-data/graphrnn_graphs/*.dat"))

    sequence_freqs_path = "/root/mlsys24-artifact/proteus/sequence_freqs.pkl"
    opcode_freqs_path = "/root/mlsys24-artifact/proteus/opcode_freqs.pkl"
