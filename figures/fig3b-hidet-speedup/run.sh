#!/bin/bash

# list of models to be obtained from torchvision
MODELS="alexnet \
densenet121 \
inception_v3 \
mnasnet0_5 \
mobilenet_v2 \
resnet50 \
vgg11 \
resnext50_32x4d"

# language models are imported from ONNX files
ONNX_DIR=/mnt/big/ybgao/mlsys24-artifact-data/onnx_models
ONNX_FILES="bert-base-uncased.onnx \
roberta-base.onnx \
distilroberta-base.onnx"

# run 8 repetitions per model
REPS=1

# use 16 nodes per partition
PARTSIZE=16

# measure torchvision speedups
for model_name in $MODELS; do
    for rep in $(seq $REPS); do
        python decompose_hidet_graph.py --model $model_name --partsize $PARTSIZE --outfile hidet_partition.csv;
    done
done

# measure language model speedups
for onnx_name in $ONNX_FILES; do
    for rep in $(seq $REPS); do
        python decompose_hidet_graph.py --onnx $ONNX_DIR/$onnx_name --partsize $PARTSIZE --outfile hidet_partition.csv;
    done
done
