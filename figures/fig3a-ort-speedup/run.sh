#!/bin/bash

mkdir figures
python split_and_profile.py /root/mlsys24-artifact-data/onnx_models/ figures/
python consolidate_results.py figures/
python draw_slowdown_plot.py