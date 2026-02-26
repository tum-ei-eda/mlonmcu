import sys
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------
# Configuration
# ---------------------------

metrics = [
    "Total Instructions",
    "ROM read-only",
    "ROM code",
    "RAM zero-init data",
    "RAM heap",
    "RAM stack",
]

backend_colors = {
    "tvmaot": "tab:blue",
    "tvmaot_tuned": "tab:blue",
    "tvmaotplus": "tab:orange",
    "tvmaotplus_tuned": "tab:orange",
    "tvmllvm": "tab:purple",
    "tvmllvm_tuned": "tab:purple",
    "emx": "tab:green",
    "ireevmvx": "tab:pink",
    "ireevmvx_inline": "tab:pink",
    "ireellvm": "tab:olive",
    "ireellvm_inline": "tab:olive",
    "ireellvmc": "tab:cyan",
    "ireellvmc_inline": "tab:cyan",
}

backend_hatches = {
    "tvmaot": None,
    "tvmaot_tuned": "///",
    "tvmaotplus": None,
    "tvmaotplus_tuned": "///",
    "tvmllvm": None,
    "tvmllvm_tuned": "///",
    "emx": None,
    "ireevmvx": None,
    "ireevmvx_inline": None,
    "ireellvm": None,
    "ireellvm_inline": None,
    "ireellvmc": None,
    "ireellvmc_inline": None,
}

MODEL_NAMES = {
    "gtsrb_cnn_supernet_preprocessed": "gtsrb_cnn_supernet (fp32)",
    "gtsrb_cnn_supernet_preprocessed_quant_static_qoperator": "gtsrb_cnn_supernet (int8)",
    "test_onnx09_mnist_simplified": "mnist_simplified",
    "test_onnx01_add_bias_tiny": "add_bias_tiny",
    "test_onnx03_conv2d_tiny": "conv2d_tiny",
    "test_onnx06_matmul_tiny.onnx": "matmul_tiny",
}

filter_models = ["gtsrb_cnn_supernet_preprocessed", "gtsrb_cnn_supernet_preprocessed_quant_static_qoperator"]
filter_backends = ["emx", "tvmaotplus", "tvmaotplus_tuned", "tflmi", "ireellvm", "ireellvmc", "ireellvm_inline", "ireellvmc_inline", "tvmllvm", "tvmllvm_tuned"]

# ---------------------------
# Input handling
# ---------------------------

if len(sys.argv) < 2:
    print("Usage: python plot_script.py <report1.csv> <report2.csv> ...")
    sys.exit(1)

csv_files = sys.argv[1:]

# ---------------------------
# Load & Merge CSVs
# ---------------------------

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Optional: remove duplicates
df = df.drop_duplicates()


def parse_config(x):
    if isinstance(x, dict):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return {}


df["Config"] = df["Config"].apply(parse_config)


def detect_autotuned(row):
    backend = row["Backend"]
    config = row["Config"]

    key = f"{backend}.ms_db"

    if key in config:
        if config[key]:
            return True
    else:
        if backend.startswith("tvm"):
            return True
    return False


df["Autotuned"] = df.apply(detect_autotuned, axis=1)
df["Backend"] = df.apply(lambda row: f"{row.Backend}_tuned" if row.Autotuned else row.Backend, axis=1)

if filter_backends:
    df = df[df["Backend"].isin(filter_backends)]

# ---------------------------
# Model processing
# ---------------------------
if filter_models:
    df = df[df["Model"].isin(filter_models)]

df["Model"] = df["Model"].apply(lambda x: MODEL_NAMES.get(x, x))

models = sorted(df["Model"].unique())
backends = sorted(df["Backend"].unique())

print(df)
# input()

x = np.arange(len(models))
width = 0.8 / len(backends)  # auto-scale width

# ---------------------------
# Plotting
# ---------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    if metric not in df.columns:
        continue

    for i, backend in enumerate(backends):
        values = []

        for model in models:
            temp = df[(df["Backend"] == backend) & (df["Model"] == model)]

            if len(temp) > 0:
                values.append(temp[metric].mean())
            else:
                values.append(0)

        ax.bar(
            x + (i - len(backends) / 2) * width + width / 2,
            values,
            width=width * 0.8,
            label=backend,
            color=backend_colors.get(backend, "gray"),
            hatch=backend_hatches.get(backend, None),
            alpha=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.5)

# Remove unused subplots if fewer metrics
for j in range(len(metrics), len(axes)):
    fig.delaxes(axes[j])

# Global legend (bottom, clean layout)
handles, labels = axes[0].get_legend_handles_labels()
if True:
    autotuned_patch = mpatches.Patch(
        facecolor="white",
        hatch="///",
        edgecolor="black",
        label="Autotuned",
    )
    handles.append(autotuned_patch)
    labels.append("Autotuned")
handles_ = []
labels_ = []
for i in range(len(handles)):
    if "_tuned" in labels[i]:
        continue
    handles_.append(handles[i])
    labels_.append(labels[i])

fig.legend(
    handles_,
    labels_,
    # loc="lower center",
    loc="upper center",
    ncol=len(labels_),
    frameon=False,
    # bbox_to_anchor=(0.5, -0.01),
    bbox_to_anchor=(0.5, 0.95),
)

fig.suptitle("Benchmark Overview", fontsize=16)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

plt.savefig("benchmark_overview.png", dpi=300)
plt.close(fig)

print("Plot generated: benchmark_overview.png")
