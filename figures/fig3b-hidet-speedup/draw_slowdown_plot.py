
import pickle
import sys
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
from scipy.stats.mstats import gmean

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Palatino",
})

plot_selection = [
    ("mobilenet", 20),
    ("resnet", 22),
    ("densenet", 53),
    # ("googlenet", 11),
    ("resnext", 22),

    ("bert", 39),
    ("roberta", 40),
    ("distilbert", 21),
]

model_name_lookup = {
    "alexnet": "alexnet",
    "densenet121": "densenet",
    "googlenet": "googlenet",
    "inception_v3": "inception",
    "mobilenet_v2": "mobilenet",
    "mobilenet_v3_large": "mobilenet",
    "resnet50": "resnet",
    "resnext50_32x4d": "resnext",
    "mnasnet0_5": "mnasnet",
    "vgg11": "vgg",
    # "shufflenet_v2_x2_0_1_3_224_224": "shufflenet",
    "/root/mlsys24-artifact-data/onnx_models/bert-base-uncased.onnx": "bert",
    "/root/mlsys24-artifact-data/onnx_models/distilroberta-base.onnx": "distilbert",
    "/root/mlsys24-artifact-data/onnx_models/roberta-base.onnx": "roberta"
}

name_order = [
    "alexnet",
    "inception",
    "mobilenet",
    "resnet",
    "densenet",
    "mnasnet",
    "mobilenet",
    "googlenet",
    "resnext",
    "vgg",
    "bert", 
    "roberta",
    "distilbert"
]

plot_data = pd.read_csv("hidet_partition.csv", names=["model type", "num_parts", "unoptimized", "best", "proteus"])
plot_data = plot_data.groupby(["model type", "num_parts"]).agg(gmean).reset_index()
plot_data["model type"] = plot_data["model type"].map(lambda k: model_name_lookup[k] if k in model_name_lookup else k)
plot_data["order"] = plot_data["model type"].map(lambda m: name_order.index(m))
plot_data = plot_data.sort_values("order", ascending=True).reset_index()
del plot_data["num_parts"], plot_data["order"], plot_data["index"]

# convert time unit
for l in ["proteus", "best", "unoptimized"]:
    plot_data[l] = 1000 * plot_data[l]

print(plot_data)

df = plot_data.copy()
df["slowdown"] = df["proteus"] / df["best"]
print(df)

plot_data.loc[len(plot_data)] = {
    "model type": "Geomean",
    "unoptimized": gmean(plot_data["unoptimized"]),
    "best": gmean(plot_data["best"]),
    "proteus": gmean(plot_data["proteus"])
}

# ax = plot_data.plot.bar(x="model type", figsize=(6, 4))
ax = plot_data.plot.bar(x="model type", figsize=(4, 2))

for idx, bar in enumerate(ax.patches):
    # Get the height (value) of the bar
    height = bar.get_height()

    model_idx = idx % len(plot_data)
    proteus_time = plot_data.iloc[model_idx]["proteus"]
    best_time = plot_data.iloc[model_idx]["best"]
    slowdown = proteus_time / best_time
    
    print(idx, bar.get_facecolor())
    
    if idx < len(plot_data):
        c = (0.3, 0.3, 0.3, 0.5)
        bar.set_facecolor(c)
    elif idx < 2 * len(plot_data):
        c = (0.196, 0.804, 0.196, 0.8)
        bar.set_facecolor(c)
    else:
        print(bar.get_x(), bar.get_y(), height)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(bar.get_x() - 1.5 * bar.get_width(), height + 5000, 
                f"${slowdown:.2f}\\times$", fontsize=8, verticalalignment='top', bbox=props)
    
    if (idx+1) % len(plot_data) == 0:
        c = list(bar.get_facecolor())
        bar.set_edgecolor(c)
        c[-1] = 0.5
        bar.set_facecolor(c)

plt.xticks(rotation=20)
plt.ylabel("runtime (ns)")
custom_labels = ["Unoptimized", "Best Attainable", "Proteus"]
ax.legend(custom_labels)
# plt.show()
plt.savefig("hidet_speedups.pdf")
