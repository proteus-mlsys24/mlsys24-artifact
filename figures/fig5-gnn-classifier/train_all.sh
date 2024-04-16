#!/bin/bash

for MODEL in resnet densenet googlenet mnasnet inception bert distilbert xlm roberta gpt2 mobilenet; do
    mkdir figures/${MODEL};
    python filter_and_train.py $MODEL --figure_dir figures/${MODEL};

    mkdir figures/${MODEL}_randop;
    python filter_and_train.py $MODEL --randomize_opcode --figure_dir figures/${MODEL}_randop;
done
