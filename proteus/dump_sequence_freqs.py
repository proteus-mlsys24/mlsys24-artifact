
import pickle
from glob import glob
from tqdm import tqdm

from proteus.graph import Graph
from proteus.graph.statistics.pathprob import opcode_frequency, opcode_sequences, list_to_frequency, merge_frequencies

onnx_paths = [
    "/home/ybgao/research/Private-Compiler/private-compiler/private_compiler/data/onnx_cache/*.onnx",
    "/home/ybgao/research/Private-Compiler/private-compiler/private_compiler/data/timm_onnx_new/*.onnx"
]

onnx_files = []
for p in onnx_paths:
    tmp = list(glob(p))
    onnx_files.extend(tmp)

sequence_freqs = dict()
opcode_freqs = dict()

for file in tqdm(onnx_files):
    G = Graph.from_onnx(file)

    # get sequence frequencies
    sequences = opcode_sequences(G, 3)
    frequencies = list_to_frequency(sequences)
    sequence_freqs = merge_frequencies(sequence_freqs, frequencies)

    # get opcode frequencies
    opcode_freqs = merge_frequencies(opcode_freqs, opcode_frequency(G))

with open("sequence_freqs.pkl", "wb") as fp:
    pickle.dump(sequence_freqs, fp)

with open("opcode_freqs.pkl", "wb") as fp:
    pickle.dump(opcode_freqs, fp)
