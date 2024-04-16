
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from proteus.config import ProteusConfig

from torch_geometric.nn import global_mean_pool, GCNConv, SAGEConv, Linear
from typing import Dict
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

class GCN(pl.LightningModule):
    def __init__(self, input_dims, hidden_dims, figure_dir=None):
        super(GCN, self).__init__()

        self.figure_dir = figure_dir
        self.opcode_mapping: Dict[str, int] = dict()
        print("GCN: input_dims:", input_dims, "hidden_dims:", hidden_dims)

        self.emb = nn.Linear(input_dims, hidden_dims)
        # self.c1 = GCNConv(hidden_dims, hidden_dims)
        self.c1 = SAGEConv(hidden_dims, hidden_dims)
        # self.c1 = GCNConv(hidden_dims, hidden_dims)
        # self.c1 = GCNConv(input_dims, hidden_dims)
        # self.c2 = GCNConv(hidden_dims, hidden_dims)
        self.c2 = SAGEConv(hidden_dims, hidden_dims)
        self.c3 = SAGEConv(hidden_dims, hidden_dims)
        self.fc = Linear(hidden_dims, 1)

        # self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()

        self.records = {
            "train": (list(), list()),
            "val": (list(), list()),
        }

        self.pos = []
        self.neg = []

    def forward(self, x, edge_index):
        # compute node embedding
        x = self.emb(x)
        x = F.relu(x)
        # graph convolutions
        x = self.c1(x, edge_index)
        x = F.relu(x)
        x = self.c2(x, edge_index)
        x = F.relu(x)
        x = self.c3(x, edge_index)
        x = F.relu(x)

        # classification
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        x = F.sigmoid(x)
        # x = F.softmax(x, dim=0)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        graph, y = batch
        pred = self(graph.x, graph.edge_index)
        l = self.loss(pred, y)
        # print(y, pred)

        for y_val, pred_val in zip(y.tolist(), pred.tolist()):
            if y_val < 0.5:
                self.records["train"][0].append(pred_val)
            else:
                self.records["train"][1].append(pred_val)

        return l

    def validation_step(self, batch, batch_idx):
        graph, y = batch
        pred = self(graph.x, graph.edge_index)
        l = self.loss(pred, y)

        for y_val, pred_val in zip(y.flatten().tolist(), pred.flatten().tolist()):
            if y_val < 0.5:
                self.records["val"][0].append(pred_val)
            else:
                self.records["val"][1].append(pred_val)

        return l


    def on_validation_epoch_end(self):
        """
        print("\n")
        print("+", np.average(self.pos), np.std(self.pos))
        print("-", np.average(self.neg), np.std(self.neg))
        print()
        """

        plt.figure(figsize=(9, 4))
        for idx, record_name in enumerate(self.records):
            pos, neg = self.records[record_name]

            total, correct = 0, 0
            true_neg, false_neg = 0, 0
            true_pos, false_pos = 0, 0
            for x in pos:
                if x < 0.5:
                    correct += 1
                    true_pos += 1
                else:
                    false_pos += 1
                total += 1
            for x in neg:
                if x > 0.5:
                    true_neg += 1
                    correct += 1
                else:
                    false_neg += 1

                total += 1

            plt.subplot(1, 2, idx+1)
            if total == 0:
                plt.title(f"{record_name}")
            else:
                plt.title(f"{record_name} ({correct}/{total}$\\approx${correct/total:.3f})")

            plt.hist(pos, bins=50, label=f"$+$ (real, n={len(pos)})", alpha=0.8)
            plt.hist(neg, bins=50, label=f"$-$ (fake, n={len(neg)})", alpha=0.8)
            plt.legend()

        plt.tight_layout()
        if self.figure_dir is None:
            plt.savefig("/tmp/predictions.pdf")
        else:
            plt.savefig(os.path.join(self.figure_dir, f"predictions.pdf"))
            
            with open(os.path.join(self.figure_dir, "stats.log"), "w") as fp:
                fp.write(f"true_neg {true_neg}\n")
                fp.write(f"false_neg {false_neg}\n")
                fp.write(f"true_pos {true_pos}\n")
                fp.write(f"false_pos {false_pos}\n")

            with open(os.path.join(self.figure_dir, "predictions.json"), "w") as fp:
                json.dump({"neg": self.records["val"][0], "pos": self.records["val"][1]}, fp)

        plt.close()


        for record_name in self.records:
            self.records[record_name][0].clear()
            self.records[record_name][1].clear()