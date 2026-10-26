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

from pathlib import Path

import numpy as np


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
