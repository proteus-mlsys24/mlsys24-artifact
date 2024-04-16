FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y vim git graphviz unzip wget python3 python3-virtualenv openssh-client

WORKDIR /root

# fetch project 
RUN mkdir .ssh && \
    ssh-keyscan -H github.com > ~/.ssh/known_hosts && \
    git clone https://github.com/proteus-mlsys24/mlsys24-artifact.git && \
    wget https://www.cs.toronto.edu/~ybgao/mlsys24-artifact-data.zip && \
    unzip mlsys24-artifact-data.zip && \
    ln -s /root/mlsys24-artifact-data mlsys24-artifact/data && \
    rm mlsys24-artifact-data.zip 

# install dependencies
RUN virtualenv venv && \
    source venv/bin/activate && \
    pushd mlsys24-artifact && \
    pip install -e . && \
    pip install scipy && \
    pip install scikit-learn community pandas && \ 
    pip install onnxruntime-gpu==1.12.1 onnx==1.12.0 && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip install hidet==0.2.4 \
    pip install lightning torch_geometric \
    popd
