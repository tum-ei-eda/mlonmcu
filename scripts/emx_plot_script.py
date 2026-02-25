import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Files for each TVM version
files = {
    # "TVM 0.11": "report_tvm_0_11.csv",
    # "TVM 0.14": "report_tvm_0_14.csv",
    # "TVM 0.17": "report_tvm_0_17.csv",
    # "TVM 0.18": "report_tvm_0_18.csv",
    # "TVM 0.18 (PvK Fork)": "report_tvm_0_18_pvk.csv",
    # "TVM 0.18 (PvK Fork)": "report_rv32gc_tvm_0_18_pvk.csv",
    "TVM 0.18 (PvK Fork)": "report_rv32gc_tvm_0_18_pvk_mnist.csv",
}

# Metrics to plot
metrics = ["Total Instructions", "ROM read-only", "ROM code", "RAM zero-init data", "RAM heap", "RAM stack"]

# Colors for backends
backend_colors = {
    "tvmaot": "tab:blue",
    "tvmaotplus": "tab:orange",
    "tvmllvm": "tab:purple",
    "emx": "tab:green",
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

for version, file in files.items():
    # Load CSV
    df = pd.read_csv(file)

    # Create model group column
    # df["ModelGroup"] = df["Model"].apply(
    #     lambda x: "Quantized" if "quant" in x.lower() else "Unquantized"
    # )
    if filter_models:
        df = df[df["Model"].isin(filter_models)]

    df["Model"] = df["Model"].apply(lambda x: MODEL_NAMES.get(x, x))
    models = df["Model"].unique()
    df["ModelGroup"] = df["Model"]
    print("df", df)
    # model = models[0]
    backends = sorted(df["Backend"].unique())
    # groups = ["Quantized", "Unquantized"]
    groups = models

    x = np.arange(len(groups))  # group positions
    width = 0.2  # width per backend bar

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for i, backend in enumerate(backends):
            values = []
            for group in groups:
                temp = df[(df["Backend"] == backend) & (df["ModelGroup"] == group)]
                if metric in temp:
                    val = temp[metric].mean()
                    values.append(val)
                else:
                    values.append(0)

            ax.bar(
                x + (i - len(backends) / 2) * width + width / 2,
                values,
                width=width * 0.8,
                label=backend,
                color=backend_colors.get(backend, "gray"),
                alpha=0.7,
            )

        ax.set_xticks(x)
        # ax.set_xticklabels(groups)
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Single legend for entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(backends), bbox_to_anchor=(0.5, 0.95))

    # fig.suptitle(f"{version} Benchmark Results", fontsize=16)
    # fig.suptitle(f"Model: gtsrb_cnn_supernet, {version}", fontsize=16)
    # fig.suptitle(f"Model: {model}, Arch: rv32gc, {version}", fontsize=16)
    fig.suptitle(f"Arch: rv32gc, {version}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # plt.savefig(f"{version.replace(' ', '_')}_benchmark.png")
    plt.savefig(f"rv32gc_{version.replace(' ', '_')}_benchmark.png")
    plt.close(fig)

print("Grouped plots generated successfully!")
