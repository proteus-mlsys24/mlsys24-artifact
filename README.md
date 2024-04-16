# Proteus MLSys 2022 Artifact

## Prerequisites
1. **Docker**

    Install Docker by following instructions at https://docs.docker.com/engine/install/.

2. **nvidia-docker** 

    To use Docker with NVIDIA GPUs, we need to install nvidia-docker. To do so, follow the instruction at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.

## Installation
1. Clone and enter the GitHub repository
    ```
    $ git clone https://github.com/jimgao1/mlsys24-artifact.git
    $ cd mlsys24-artifact/
    ```
2. Build the Docker container
    ```
    $ docker build -t mlsys24-artifact docker/
    ```
3. Run the Docker container with GPU
    ```
    $ docker run --gpus 0 --privileged -it mlsys24-artifact
    ```

## Figure Reproduction
* Figure 3a: ONNXRuntime Speedup

    Navigate to `figures/fig3a-ort-speedup` and run `run.sh`.
* Figure 3b: Hidet Speedup

    Navigate to `figures/fig3b-hidet-speedup` and run `run.sh`.
* Figure 5: GNN Classifier Adversary

    Navigate to `figures/fig5-gnn-classifier` and follow the instructions in README.md