
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
    ("mobilenet", 11),
    ("resnet", 10),
    ("densenet", 19),
    ("googlenet", 11),
    ("resnext", 10),

    ("bert", 16),
    ("roberta", 16),
    ("distilbert", 16),
]

model_name_lookup = {
    "densenet121_1_3_224_224": "densenet",
    "googlenet_1_3_224_224": "googlenet",
    "inception_v3_1_3_224_224": "inception",
    "mobilenet_v3_large_1_3_224_224": "mobilenet",
    "resnet50_1_3_224_224": "resnet",
    "resnext50_32x4d_1_3_224_224": "resnext",
    "shufflenet_v2_x2_0_1_3_224_224": "shufflenet",
    "bert_shapes": "bert",
    "distilbert_shapes": "distilbert",
    "roberta_shapes": "roberta"
}

slowdown_data = []

for f in ["slowdown_data.pkl"]:
    with open(f, "rb") as fp:
        tmp = pickle.load(fp)
        slowdown_data.extend(tmp)

print(slowdown_data)

parts_available = dict()
for model, nparts, reassembled, optimized, speedup in slowdown_data:
    if model not in model_name_lookup:
        print(f"Warning: {model} not found in lookup.")
        continue
    model = model_name_lookup[model]
    if model not in parts_available:
        parts_available[model] = []
    parts_available[model].append(nparts)

for k,v in parts_available.items():
    print(k, v)

# plot_data contains ["model type", "unoptimized", "best", "proteus"]
# slowdown_data contains ('googlenet_1_3_224_224', 1, 6254.289128974008, 4530.6575343500435, 1.3804374048481756)

time_unoptimized = dict()
time_best = dict()
time_proteus = dict()

for model, nparts, reassembled, optimized, speedup in slowdown_data:
    if model not in model_name_lookup: continue
    model = model_name_lookup[model]
    if nparts == 1:
        time_unoptimized[model] = reassembled
        time_best[model] = optimized
    
    if (model, nparts) in plot_selection:
        time_proteus[model] = optimized

rows = []
for k, _ in plot_selection:
    if k not in time_proteus: continue
    print(k, time_unoptimized[k], time_best[k], time_proteus[k])
    rows.append([k, time_unoptimized[k], time_best[k], time_proteus[k]])

plot_data = pd.DataFrame.from_records(rows, columns=["model type", "unoptimized", "best", "proteus"])

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
        ax.text(bar.get_x() - 1.5 * bar.get_width(), height + 1500, 
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
plt.savefig("speedups.pdf")