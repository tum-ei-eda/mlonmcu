"""Generate raw TFLite golden outputs without replacing existing validation data.

Example: python scripts/generate_validation_references.py workspace_default/models/toycar --number 10
The destination defaults to output_tflite_generated; update definition.yml to select it.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml


def generate(model_dir, number, output_dir):
    import tensorflow as tf

    metadata = yaml.safe_load((model_dir / "definition.yml").read_text())
    nodes = metadata["network_parameters"]
    interpreter = tf.lite.Interpreter(model_path=str(model_dir / metadata["network"]["filename"]))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != len(nodes["input_nodes"]) or len(outputs) != len(nodes["output_nodes"]):
        raise ValueError("Metadata tensor count does not match the model")
    if number <= 0:
        raise ValueError("number must be positive")
    # Refuse to overwrite an earlier reference set, including partially generated data.
    output_dir.mkdir(parents=True, exist_ok=False)
    for sample in range(number):
        for index, (tensor, node) in enumerate(zip(inputs, nodes["input_nodes"])):
            directory = model_dir / node["example_input"]["path"]
            filename = f"{sample}.bin" if len(inputs) == 1 else f"{sample}_{index}.bin"
            storage_type = node.get("quantize", {}).get("dtype", node["dtype"])
            data = np.frombuffer((directory / filename).read_bytes(), dtype=storage_type).reshape(tensor["shape"])
            dtype = np.dtype(tensor["dtype"])
            if data.dtype != dtype and np.issubdtype(dtype, np.integer):
                scale, zero = tensor["quantization"]
                if scale <= 0:
                    raise ValueError("Missing per-tensor input quantization")
                limits = np.iinfo(dtype)
                data = np.clip(np.around(data / scale + zero), limits.min, limits.max)
            interpreter.set_tensor(tensor["index"], data.astype(dtype))
        interpreter.invoke()
        for index, (tensor, node) in enumerate(zip(outputs, nodes["output_nodes"])):
            data = interpreter.get_tensor(tensor["index"])
            storage_type = node.get("dequantize", {}).get("dtype", node["dtype"])
            if np.issubdtype(data.dtype, np.integer) and np.issubdtype(np.dtype(storage_type), np.floating):
                scale, zero = tensor["quantization"]
                data = (data.astype("float32") - zero) * scale
            filename = f"{sample}.bin" if len(outputs) == 1 else f"{sample}_{index}.bin"
            (output_dir / filename).write_bytes(data.astype(storage_type).tobytes())
        if (sample + 1) % 1000 == 0 or sample + 1 == number:
            print(f"Generated {sample + 1}/{number} samples in {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    generate(args.model_dir, args.number, args.output_dir or args.model_dir / "output_tflite_generated")


if __name__ == "__main__":
    main()
