# GNN Classifier Adversary
\# | Stage | Approximate time on testing setup
---| ----- | -----------------------------
1 | Populate distributions | 5-10 minutes
2 | Real subgraph extraction | 25 minutes
3 | Sentinel subgraph generation | 5-8 hours
4 | Consolidate fakes | 10 minutes
5 | Model training | <1 hour
6 | Tabulation | < 1 minute

## Option 1: Use pre-generated real + fake graphs
1. Copy the pre-generated real and fake subgraphs
    ```
    cp /root/mlsys24-artifact-data/pre_generated_sentinels/* .
    ```
2. Run `runall.sh`: For each DNN class, trains the GNN and returns the GNN-reduced search space. 
    ```
    $ ./train_all.sh
    ```
3. Run `tabulate_results.py`: Creates the table containing the data in figure 5.
    ```
    $ python tabulate_results.py figures/
    ```

## Option 2: Run the whole pipeline
This might take a long time:
1. Run `populate_distributions.py`: Generates the library of graphs from GraphRNN with a specified number of inputs and outputs. 
    ```
    $ python populate_distributions.py ~/mlsys24-artifact-data/graphrnn_graphs --nthreads 32
    ```
2. Run `generate_reals.py`: Partitions ONNX graphs and generates the `reals.pkl` file containing fragments of real topologies.
    ```
    $ python generate_reals.py --onnx_dir /root/mlsys24-artifact/data/onnx_models
    ```
3. Run `generate_fakes.py`: Based on the real topologies, generate a number of folders containing fake graphs. 
    ```
    $ python generate_fakes.py
    $ python generate_fakes.py --num_mutants 2
    ```
4. Run `consolidate_fakes.py`: Combines the results from the folders and generates a `consolidated_fakes.pkl` file. 
    ```
    $ python consolidate_fakes.py
    ```
5. Run `runall.sh`: For each DNN class, trains the GNN and returns the GNN-reduced search space. 
    ```
    $ ./train_all.sh
    ```
6. Run `tabulate_results.py`: Creates the table containing the data in figure 5.
    ```
    $ python tabulate_results.py figures/
    ```



