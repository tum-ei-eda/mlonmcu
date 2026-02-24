import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mlonmcu.config import str2list, str2dict


# TODO: explore hill_climb algos and disable_legalize

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
    "tflmi": "tab:brown",
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
    "tflmi": None,
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
    "micro_kws_m_fp32_static.tosa": "micro_kws_m_fp32",
    "micro_kws_m_quantized_static.tosa": "micro_kws_m_quantized",
    "micro_kws_m_static_fp32": "micro_kws_m_fp32",
    "micro_kws_m_static_qoperator_emx": "micro_kws_m_quantized",
}

"""
    Run                                 Model Frontend  ... Failing                  Reason Autotuned
0     0                      micro_kws_m_fp32   tflite  ...     NaN                     NaN     False
1     1                 micro_kws_m_quantized   tflite  ...     NaN                     NaN     False
2     2               micro_kws_m_static_fp32     onnx  ...     NaN                     NaN     False
3     3      micro_kws_m_static_qoperator_emx     onnx  ...     NaN                     NaN     False
4     4          micro_kws_m_static_qoperator     onnx  ...    True  AssertionError @ BUILD     False
5     5      micro_kws_m_static_qoperator_tvm     onnx  ...     NaN                     NaN     False
6     6  micro_kws_m_static_quantized_dequant     onnx  ...     NaN                     NaN     False
7     7          micro_kws_m_static_quantized     onnx  ...     NaN                     NaN     False
8     0      micro_kws_m_static_qoperator_emx     onnx  ...     NaN                     NaN     False
9     1               micro_kws_m_static_fp32     onnx  ...     NaN                     NaN     False
10    2  micro_kws_m_static_quantized_dequant     onnx  ...     NaN                     NaN     False
"""

# filter_models = ["gtsrb_cnn_supernet_preprocessed", "gtsrb_cnn_supernet_preprocessed_quant_static_qoperator"]
filter_models = ["micro_kws_m_fp32", "micro_kws_m_quantized", "micro_kws_m_static_fp32", "micro_kws_m_static_qoperator_emx", "micro_kws_m_quantized_static.tosa", "micro_kws_m_fp32_static.tosa"]
filter_backends = ["emx", "tvmaotplus", "tvmaotplus_tuned", "tflmi", "ireellvm", "ireellvmc", "ireellvm_inline", "ireellvmc_inline", "tvmllvm", "tvmllvm_tuned", "tvmaot", "tvmaot_tuned"]

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

import ast

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

def detect_attrs(row):
    backend = row["Backend"]
    config = row["Config"]
    features = row["Features"]
    attrs = []

    # desired_layout
    key = f"{backend}.desired_layout"
    val = config.get(key)
    if val is not None:
        attr = val
        attrs.append(attr)

    # target_device
    key = f"{backend}.target_device"
    val = config.get(key)
    if val is not None:
        attr = val
        attrs.append(attr)

    # disable_legalize
    key = "disable_legalize"
    if key in features:
        attr = key
        attrs.append(attr)

    # disabled_passes
    key = f"{backend}.disabled_passes"
    val = config.get(key)
    if val is not None:
        if isinstance(val, str):
            val = str2list(val)
        assert isinstance(val, list)
        for val_ in val:
            attr = f"No{val_}"
            attrs.append(attr)

    # extra_pass_config
    key = f"{backend}.extra_pass_config"
    val = config.get(key)
    if val is not None:
        if isinstance(val, str):
            val = str2dict(val)
        assert isinstance(val, dict)
        for k, v in val.items():
            # attr = f"{k}={v}"
            attr = f"{v}"
            attrs.append(attr)

    attrs_str = ",".join(attrs)

    return attrs_str

df["Autotuned"] = df.apply(detect_autotuned, axis=1)
df["Attrs"] = df.apply(detect_attrs, axis=1)
df["Backend"] = df.apply(lambda row: f"{row.Backend}_tuned" if row.Autotuned else row.Backend, axis=1)


# ---------------------------
# Model processing
# ---------------------------
if filter_models:
    df = df[df["Model"].isin(filter_models)]

if filter_backends:
    df = df[df["Backend"].isin(filter_backends)]

df["Model"] = df["Model"].apply(lambda x: MODEL_NAMES.get(x, x))
# df["Model"] = df.apply(lambda row: f"{row.Model}.{row.Frontend}" ,axis=1)


AGGREGATE_METRICS = True
def make_backend_variant(row, df_grouped):
    """
    row: single row of df
    df_grouped: all rows of same (Model, Backend)
    returns list of BackendVariant(s)
    """
    base_backend = row["Backend"]

    if not AGGREGATE_METRICS:
        # default behavior: include attribute if present
        return base_backend if not row["Attrs"] else f"{base_backend} [{row['Attrs']}]"

    # AGGREGATE_METRICS=True
    # If only one attribute combination, keep normal
    if len(df_grouped) == 1:
        return base_backend

    # multiple attrs → create min/mean/max variants
    return [
        f"{base_backend} [min]",
        f"{base_backend} [mean]",
        f"{base_backend} [max]",
    ]

# df["BackendVariant"] = df.apply(
#     lambda row: row["Backend"] if not row["Attrs"]
#     else f"{row['Backend']} [{row['Attrs']}]",
#     axis=1
# )
if AGGREGATE_METRICS:
    # Pre-group by Model + Backend
    df_variants = []
    for (model, backend), group in df.groupby(['Model', 'Backend']):
        metric_cols = [m for m in metrics if m in group.columns]
        if AGGREGATE_METRICS and len(group) > 1:
            # Compute min/mean/max per metric
            agg_values = {
                "min": group[metric_cols].min(),
                "mean": group[metric_cols].mean(),
                "max": group[metric_cols].max(),
            }
    
            # Create one row per aggregation
            for stat in ["min", "mean", "max"]:
                row = group.iloc[0].copy()  # take first row as template
                row["BackendVariant"] = f"{backend} [{stat}]"
                row["Attrs"] = stat
    
                for col in metric_cols:
                    row[col] = agg_values[stat][col]
    
                df_variants.append(row)
        else:
            # Only one attribute, keep original row
            row = group.iloc[0].copy()
            # row["BackendVariant"] = backend if not row["Attrs"] else f"{backend} [{row['Attrs']}]"
            row["BackendVariant"] = f"{backend} [mean]"
            df_variants.append(row)

    df = pd.DataFrame(df_variants)

else:
    # Original behavior: one BackendVariant per row
    df["BackendVariant"] = df.apply(
        lambda row: row["Backend"] if not row["Attrs"]
        else f"{row['Backend']} [{row['Attrs']}]",
        axis=1
    )

models = sorted(df["Model"].unique())
# backends = sorted(df["Backend"].unique())
backends = sorted(df["BackendVariant"].unique())

TO_DROP = ["Run", "Platform", "Comment", "Reason", "Failing", "Kernels Size", "Postprocesses", "Config", "Load Stage Time [s]", "Build Stage Time [s]", "Compile Stage Time [s]", "Run Stage Time [s]", "Workspace Size [B]", "Features"]
for col in TO_DROP:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

print(df)
print("backends", backends)
# input()

x = np.arange(len(models))
width = 0.8 / len(backends)  # auto-scale width

# ---------------------------
# Plotting
# ---------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# fig, axes = plt.subplots(2, 3, figsize=(18*2, 10*2))
# fig, axes = plt.subplots(2, 3, figsize=(18*4, 10*4))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    if metric not in df.columns:
        continue

    for i, backend in enumerate(backends):
        values = []

        for model in models:
            # temp = df[
            #     (df["Backend"] == backend) &
            #     (df["Model"] == model)
            # ]
            temp = df[
                (df["BackendVariant"] == backend) &
                (df["Model"] == model)
            ]



            if len(temp) > 0:
                temp_vals = temp[metric]
                # print("temp_vals", temp_vals)
                num_rows = len(temp_vals)
                # assert num_rows == 1
                temp_val = temp_vals.mean()
                values.append(temp_val)
            else:
                values.append(0)

        base_backend = backend.split(" [")[0]
        bars = ax.bar(
            x + (i - len(backends)/2) * width + width/2,
            values,
            width=width*0.8,
            label=backend,
            color=backend_colors.get(base_backend, "gray"),
            hatch=backend_hatches.get(base_backend, None),
            alpha=0.7,
        )
        for j, bar in enumerate(bars):

            model = models[j]
        
            temp = df[
                (df["BackendVariant"] == backend) &
                (df["Model"] == model)
            ]
        
            if len(temp) == 0:
                continue
        
            attrs = temp["Attrs"].iloc[0]
        
            if attrs:
                # optional shortening
                attrs_short = (
                    attrs
                    # .replace("NoAlterOpLayout", "NoLayoutOpt")
                    # .replace(",", "\n")
                )
        
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    # bar.get_height() ,
                    # 1.0-0.02,
                    0.02,
                    attrs_short,
                    ha="center",
                    # va="top",
                    va="bottom",
                    # va="center",
                    fontsize=7,
                    rotation=90,
                    clip_on=False,
                    transform=ax.get_xaxis_transform(),
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
    if "[" in labels[i] and "[mean]" not in labels[i]:
        continue
    handles_.append(handles[i])
    label = labels[i]
    label = label.replace(" [mean]", "")
    labels_.append(label)

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

