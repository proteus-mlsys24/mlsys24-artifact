import os
import json
import argparse
from glob import glob
import pandas as pd
import numpy as np

N = {
    "densenet": 19,
    "googlenet": 11,
    "inception": 19,
    "mnasnet": 11,
    "resnet": 10,
    "mobilenet": 11,
    "bert": 16,
    "roberta": 16,
    "xlm": 25,
}

K = 20

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("figure_dir", type=str)
    args = parser.parse_args()

    rows = []

    model_dirs = glob(os.path.join(args.figure_dir, "*"))
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        random_opcode = False
        if model_name.endswith("randop"):
            random_opcode = True
            model_name = model_name[:-len("_randop")]

        if not os.path.exists(os.path.join(model_dir, "predictions.json")): continue

        with open(os.path.join(model_dir, "predictions.json"), "r") as fp:
            predictions = json.load(fp)

        total = len(predictions["pos"]) + len(predictions["neg"])
        correct = len(list(filter(lambda n: n>0.5, predictions["pos"]))) + len(list(filter(lambda n: n<0.5, predictions["neg"])))

        val_accuracy = correct / total

        # max_real_pred = max(predictions["neg"])
        max_real_pred = np.percentile(predictions["neg"], 99)
        specificity = len(list(filter(lambda n: n>max_real_pred, predictions["pos"]))) / len(predictions["pos"])

        num_candidates = (1 + (1 - specificity) * K) ** N[model_name]

        rows.append((model_name, random_opcode, val_accuracy, max_real_pred, specificity, num_candidates))

df = pd.DataFrame.from_records(rows, columns=["model_name", "random_opcode", "val_accuracy", "max_real_pred", "specificity", "num_candidates"]).sort_values("model_name")

g = df.groupby("model_name")
for key, item in g:
    print(g.get_group(key).sort_values("random_opcode"), "\n\n")
