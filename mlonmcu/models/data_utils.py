#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Model-independent helpers for validation inputs, references, labels, and metadata."""

import hashlib
import json
from pathlib import Path

import numpy as np
from filelock import FileLock

from mlonmcu.logging import get_logger
from mlonmcu.setup import utils


def write_model_data_manifest(model_dir, paths, destination=None):
    """Record SHA-256 hashes and sample count from a complete, trusted dataset."""
    model_dir = Path(model_dir).resolve()
    files = {}
    sample_sets = []
    for entry in paths:
        path = (model_dir / entry).resolve()
        path.relative_to(model_dir)
        if path.is_dir():
            entries = sorted(path.glob("*.bin"))
            samples = {int(item.stem.split("_", 1)[0]) for item in entries}
            if not samples or samples != set(range(len(samples))):
                raise ValueError(f"Expected contiguous samples starting at zero in {path}")
            sample_sets.append(samples)
        else:
            entries = [path]
        for item in entries:
            files[item.relative_to(model_dir).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
    if not sample_sets or any(samples != sample_sets[0] for samples in sample_sets):
        raise ValueError("Manifest data directories must describe the same samples")
    manifest = {"version": 1, "sample_count": len(sample_sets[0]), "files": files}
    destination = Path(destination) if destination is not None else model_dir / "data_manifest.json"
    with destination.open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def ensure_model_data(model_dir, input_paths=(), output_paths=(), label_paths=(), number=10):
    """Ensure requested default validation files exist, downloading only when needed.

    Callers opt in before invoking this function. A data_manifest.json supplies
    the dataset size and trusted SHA-256 hashes. Only requested files are hashed.
    Return the requested sample count, capped by the manifest's dataset size.
    """
    if model_dir is None or not (input_paths or output_paths or label_paths):
        return number
    model_dir = Path(model_dir).resolve()
    script = model_dir / "download_data.sh"
    manifest_path = model_dir / "data_manifest.json"
    hashes = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        count = manifest.get("sample_count")
        if manifest.get("version") != 1 or type(count) is not int or count <= 0:
            raise ValueError(f"Invalid version or sample_count in {manifest_path}")
        hashes = manifest.get("files")
        if not isinstance(hashes, dict):
            raise ValueError(f"Missing file hashes in {manifest_path}")
        for filename, digest in hashes.items():
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Manifest filenames must be relative to the model directory: {filename}")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid SHA-256 digest for {filename}")
        number = min(number, count)

    requirements = []
    for paths in (input_paths, output_paths):
        for tensor, directory in enumerate(paths):
            directory = Path(directory)
            for sample in range(number):
                filename = f"{sample}.bin" if len(paths) == 1 else f"{sample}_{tensor}.bin"
                alternatives = [directory / filename]
                if len(paths) == 1:
                    alternatives.append(directory / f"{sample}_0.bin")
                requirements.append(alternatives)
    requirements.extend([[Path(path)] for path in label_paths])
    if hashes is not None:
        for alternatives in requirements:
            alternatives[:] = [
                path for path in alternatives if path.resolve().relative_to(model_dir).as_posix() in hashes
            ]
            if not alternatives:
                raise ValueError(f"{manifest_path} does not describe all requested validation files")

    def valid_file(path):
        if not path.is_file() or not path.stat().st_size:
            return False
        if hashes is None:
            return True
        name = path.resolve().relative_to(model_dir).as_posix()
        return hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]

    def missing_files():
        return [paths[0] for paths in requirements if not any(valid_file(path) for path in paths)]

    missing = missing_files()
    if not missing:
        return number
    if not script.is_file():
        if hashes is not None:
            raise RuntimeError(f"Missing or invalid validation data: {missing[0]}; no download_data.sh available")
        return number
    # Parallel runs/backends can share one model directory. Recheck under the lock.
    with FileLock(str(model_dir / ".download_data.lock")):
        if not missing_files():
            return number
        get_logger().info("Downloading missing validation data using %s", script)
        # The existing model downloaders expect their extraction directories to exist.
        for path in (*input_paths, *output_paths):
            Path(path).mkdir(parents=True, exist_ok=True)
        utils.execute("bash", "-e", str(script), cwd=model_dir)
        missing = missing_files()
        if missing:
            raise RuntimeError(f"{script} did not provide required validation data: {missing[0]}")
    return number


def generate_input_data(
    input_names,
    input_types,
    input_shapes,
    input_ranges,
    input_quant_details,
    in_paths,
    *,
    fill_mode="file",
    data_file="auto",
    number=10,
):
    """Generate tensor inputs from raw files or synthetic values."""
    inputs_data = []
    if fill_mode in ["zeros", "ones", "random"]:
        for _ in range(number):
            data = {}
            for name in input_names:
                dtype = np.dtype(input_types[name])
                shape = input_shapes[name]
                # Synthetic data is generated directly in the model tensor's dtype.
                if fill_mode == "zeros":
                    arr = np.zeros(shape, dtype=dtype)
                elif fill_mode == "ones":
                    arr = np.ones(shape, dtype=dtype)
                else:
                    bounds = input_ranges.get(name)
                    if np.issubdtype(dtype, np.integer):
                        info = np.iinfo(dtype)
                        lower, upper = bounds or (info.min, info.max)
                        arr = np.random.randint(lower, int(upper) + 1, size=shape, dtype=dtype)
                    else:
                        lower, upper = bounds or (-0.5, 0.5)
                        arr = np.random.uniform(lower, upper, size=shape).astype(dtype)
                data[name] = arr
            inputs_data.append(data)
    elif fill_mode == "file":
        if data_file == "auto":
            assert len(in_paths) > 0, "in_paths is empty"
            if len(in_paths) == 1:
                if in_paths[0].is_dir():
                    files = list(in_paths[0].iterdir())
                else:
                    assert in_paths[0].is_file(), "in_paths not found"
                    files = [in_paths[0]]
            else:
                files = in_paths
            temp = {}
            NEW = True
            for file in files:
                if int(Path(file).stem.split("_", 1)[0]) >= number:
                    continue
                if not isinstance(file, Path):
                    file = Path(file)
                assert file.is_file(), f"Not found: {file}"
                basename, ext = file.stem, file.suffix
                if ext == ".bin":
                    if "_" in basename:
                        i, ii = basename.split("_", 1)
                        i = int(i)
                        ii = int(ii)
                    else:
                        i = int(basename)
                        ii = 0
                    with open(file, "rb") as f:
                        data = f.read()
                    if i not in temp:
                        temp[i] = {}
                    temp[i][ii] = data
                elif ext in [".npy", ".npz"]:
                    raise NotImplementedError
                else:
                    raise RuntimeError(f"Unsupported ext: {ext}")
            assert len(temp) > 0
            for i in range(min(number, len(temp))):
                assert i in temp
                data = {}
                for ii, input_name in enumerate(input_names):
                    assert ii in temp[i]
                    assert input_name in input_types, f"Unknown dtype for input: {input_name}"
                    dtype = input_types[input_name]
                    quant = input_quant_details.get(input_name, None)
                    rng = input_ranges.get(input_name, None)
                    gen_dtype = dtype
                    if quant:
                        _, _, ty, _ = quant
                        # assert "float" in ty, "Input already quantized?"
                        if NEW:
                            gen_dtype = ty
                    arr = np.frombuffer(temp[i][ii], dtype=gen_dtype)
                    assert input_name in input_shapes, f"Unknown shape for input: {input_name}"
                    shape = input_shapes[input_name]
                    arr = np.reshape(arr, shape)
                    # Quantize if required
                    # if gen_dtype != dtype:
                    if quant and gen_dtype != dtype:
                        assert "int" in dtype
                        # assert quant
                        scale, shift, ty, qrng = quant
                        if qrng is not None:
                            assert len(qrng) == 2, "Range should be a tuple (lower, upper)"
                            lower, upper = qrng
                            assert lower <= upper
                            CLIP_INPUTS = True
                            if CLIP_INPUTS:
                                arr = np.clip(arr, lower, upper)
                            else:
                                assert np.min(arr) >= lower or np.isclose(np.min(arr), lower), "Range missmatch (lower)"
                                assert np.max(arr) <= upper or np.isclose(np.max(arr), upper), "Range missmatch (upper)"
                        arr = (arr / scale) + shift
                        arr = np.around(arr)
                        limits = np.iinfo(dtype)
                        arr = np.clip(arr, limits.min, limits.max).astype(dtype)
                        # input("!=")
                    if rng is not None:
                        # TODO: Move shared code!
                        assert len(rng) == 2, "Range should be a tuple (lower, upper)"
                        lower, upper = rng
                        assert lower <= upper
                        CLIP_INPUTS = True
                        if CLIP_INPUTS:
                            arr = np.clip(arr, lower, upper)
                        else:
                            assert np.min(arr) >= lower or np.isclose(np.min(arr), lower), "Range missmatch (lower)"
                            assert np.max(arr) <= upper or np.isclose(np.max(arr), upper), "Range missmatch (upper)"
                    data[input_name] = arr
                inputs_data.append(data)
        else:
            assert data_file is not None, "Missing value for gen_data_file"
            file = Path(data_file)
            assert file.is_file(), f"File not found: {file}"
            raise NotImplementedError("Explicit gen_data.file is not implemented; use raw model input directories")
        # for i, input_name in enumerate(input_names):

    elif fill_mode == "dataset":
        raise NotImplementedError
    else:
        raise RuntimeError(f"unsupported fill_mode: {fill_mode}")
    return inputs_data


def generate_output_ref_data(
    inputs_data,
    model,
    out_paths,
    output_names,
    output_types,
    output_shapes,
    output_quant_details,
    *,
    mode="file",
    data_file="auto",
    inference=None,
):
    """Load golden output tensors or generate them with the supplied inference callback."""
    outputs_data = []
    if mode == "model":
        assert inference is not None, "Model reference generation requires an inference callback"
        assert len(inputs_data) > 0
        for i, input_data in enumerate(inputs_data):
            # input("321?")
            output_data = inference(model, input_data, quant=False, dequant=True)
            outputs_data.append(output_data)
            # input("321!")

    elif mode == "file":
        if data_file == "auto":
            assert len(out_paths) > 0, "out_paths is empty"
            if len(out_paths) == 1:
                if out_paths[0].is_dir():
                    files = list(out_paths[0].iterdir())
            else:
                files = out_paths
            temp = {}
            assert len(inputs_data) <= len(
                files
            ), f"Missing output data for provided inputs. (Expected: {len(inputs_data)}, Got: {len(files)})"
            for file in files:
                if int(Path(file).stem.split("_", 1)[0]) >= len(inputs_data):
                    continue
                if not isinstance(file, Path):
                    file = Path(file)
                assert file.is_file()
                basename, ext = file.stem, file.suffix
                if ext == ".bin":
                    if "_" in basename:
                        i, ii = basename.split("_", 1)
                        i = int(i)
                        ii = int(ii)
                    else:
                        i = int(basename)
                        ii = 0
                    with open(file, "rb") as f:
                        data = f.read()
                    if i not in temp:
                        temp[i] = {}
                    temp[i][ii] = data
                elif ext in [".npy", ".npz"]:
                    raise NotImplementedError
                else:
                    raise RuntimeError(f"Unsupported ext: {ext}")
            # TODO: handle case where there are more output samples than input samples?
            for i in range(len(inputs_data)):
                assert i in temp
                data = {}
                for ii, output_name in enumerate(output_names):
                    assert ii in temp[i]
                    assert output_name in output_types, f"Unknown dtype for output: {output_name}"
                    dtype = output_types[output_name]
                    dequant = output_quant_details.get(output_name, None)
                    if dequant:
                        _, _, ty, _ = dequant
                        dtype = ty
                    arr = np.frombuffer(temp[i][ii], dtype=dtype)
                    assert output_name in output_shapes, f"Unknown shape for output: {output_name}"
                    shape = output_shapes[output_name]
                    arr = np.reshape(arr, shape)
                    data[output_name] = arr
                outputs_data.append(data)
        else:
            assert data_file is not None, "Missing value for gen_ref_data_file"
            file = Path(data_file)
            assert file.is_file(), f"File not found: {file}"
            raise NotImplementedError
    else:
        raise RuntimeError(f"unsupported fill_mode: {mode}")
    return outputs_data


def generate_ref_labels(
    inputs_data,
    model,
    out_labels_paths,
    *,
    mode="file",
    data_file="auto",
    inference=None,
):
    """Load reference classification labels or infer them from model outputs."""
    labels = []
    if mode == "model":
        assert inference is not None, "Model reference generation requires an inference callback"
        assert len(inputs_data) > 0
        for i, input_data in enumerate(inputs_data):
            output_data = inference(model, input_data, quant=False, dequant=True)
            assert len(output_data) == 1, "Does not support multi-output classification"
            output_data = output_data[list(output_data)[0]]
            top_label = np.argmax(output_data)
            labels.append(top_label)

    elif mode == "file":
        if data_file == "auto":
            assert len(out_labels_paths) > 0, "labels_paths is empty"
            assert len(out_labels_paths) == 1
            file = Path(out_labels_paths[0])
        else:
            assert data_file is not None, "Missing value for gen_ref_labels_file"
            file = Path(data_file)
        assert file.is_file(), f"File not found: {file}"
        ext = file.suffix
        assert len(ext) > 1
        fmt = ext[1:].lower()
        if fmt == "csv":
            import pandas as pd

            labels_df = pd.read_csv(file, sep=",")
            assert "i" in labels_df.columns
            assert "label_idx" in labels_df.columns
            assert len(inputs_data) <= len(labels_df)
            labels_df.sort_values("i", inplace=True)
            labels = list(labels_df["label_idx"].astype(int))[: len(inputs_data)]
        else:
            raise NotImplementedError(f"Fmt not supported: {fmt}")
    else:
        raise RuntimeError(f"unsupported fill_mode: {mode}")
    return labels


def generate_model_info(
    input_names,
    output_names,
    input_shapes,
    output_shapes,
    input_types,
    output_types,
    input_ranges,
    output_ranges,
    input_quant_details,
    output_quant_details,
):
    """Build ordered tensor metadata for subsequent validation stages."""
    model_info_dict = {
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": [input_shapes.get(name) for name in input_names],
        "output_shapes": [output_shapes.get(name) for name in output_names],
        "input_types": [input_types.get(name) for name in input_names],
        "output_types": [output_types.get(name) for name in output_names],
        "input_ranges": [input_ranges.get(name) for name in input_names],
        "output_ranges": [output_ranges.get(name) for name in output_names],
        "input_quant_details": [input_quant_details.get(name) for name in input_names],
        "output_quant_details": [output_quant_details.get(name) for name in output_names],
    }
    # nested version
    # model_info_dict = {
    #     "inputs": [
    #         {
    #             "name": "input_1",
    #             "shape": [1, 1014],
    #             "type": "int8",
    #         }
    #     ],
    #     "outputs": [
    #         {
    #             "name": "output",
    #             "shape": [1, 10],
    #             "type": "int8",
    #         }
    #     ],
    # }
    return model_info_dict  # TODO: turn into class
