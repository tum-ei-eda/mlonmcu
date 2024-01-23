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
import pytest
import numpy as np

from mlonmcu.models.utils import make_hex_array, fill_data_source, lookup_data_buffers, get_data_source


def test_models_utils_make_hex_array_bin(tmp_path_factory):
    data = [0, 1, 3, 5, 16, 33, 80]
    data = bytes(data)
    data_dir = tmp_path_factory.mktemp("data")
    data_file = data_dir / "f.bin"
    with open(data_file, "wb") as f:
        f.write(data)
    out = make_hex_array(data_file, mode="bin")
    assert out.strip() == "0x00, 0x01, 0x03, 0x05, 0x10, 0x21, 0x50,"
    out = make_hex_array(data_file, mode="auto")
    assert out.strip() == "0x00, 0x01, 0x03, 0x05, 0x10, 0x21, 0x50,"


@pytest.mark.parametrize("compressed", [False, True])
def test_models_utils_make_hex_array_npy(tmp_path_factory, compressed):
    data = np.array([0, 1, -3, 5, 16, -33, -80], dtype="int8")
    data_dir = tmp_path_factory.mktemp("data")
    ext = "npz" if compressed else "npy"
    data_file = data_dir / f"f.{ext}"
    if compressed:
        np.savez(data_file, data)
    else:
        np.save(data_file, data)
    out = make_hex_array(data_file, mode=ext)
    assert out.strip() == "0x00, 0x01, 0xfd, 0x05, 0x10, 0xdf, 0xb0,"
    out = make_hex_array(data_file, mode="auto")
    assert out.strip() == "0x00, 0x01, 0xfd, 0x05, 0x10, 0xdf, 0xb0,"


def test_models_utils_make_hex_array_invalid(tmp_path_factory):
    data = [0, 1, 3, 5, 16, 33, 80]
    data = bytes(data)
    data_dir = tmp_path_factory.mktemp("data")
    data_file = data_dir / "f"
    with pytest.raises(RuntimeError):
        make_hex_array(data_file, mode="foo")  # Invalid mode
    with pytest.raises(AssertionError):
        make_hex_array(data_file, mode="auto")  # Missing extension
    data_file = data_dir / "f.bin"
    with pytest.raises(FileNotFoundError):
        make_hex_array(data_file, mode="auto")  # File does not exist
    with open(data_file, "wb") as f:
        f.write(bytes([]))
    with pytest.raises(AssertionError):
        make_hex_array(data_file, mode="auto")  # Empty data


def test_models_utils_fill_data_source_empty():
    out = fill_data_source([], [])
    assert (
        out
        == """#include "ml_interface.h"
#include <stddef.h>
const int num_data_buffers_in = 0;
const int num_data_buffers_out = 0;
const unsigned char *const data_buffers_in[] = {};
const unsigned char *const data_buffers_out[] = {};
const size_t data_size_in[] = {};
const size_t data_size_out[] = {};
"""
    )


def test_models_utils_fill_data_source():
    out = fill_data_source([["0x01"], ["0x02"], ["0x03"]], [["0x04"], ["0x05"], ["0x06"]])
    assert (
        out
        == """#include "ml_interface.h"
#include <stddef.h>
const int num_data_buffers_in = 3;
const int num_data_buffers_out = 3;
const unsigned char data_buffer_in_0_0[] = {0x01};
const unsigned char data_buffer_in_1_0[] = {0x02};
const unsigned char data_buffer_in_2_0[] = {0x03};
const unsigned char data_buffer_out_0_0[] = {0x04};
const unsigned char data_buffer_out_1_0[] = {0x05};
const unsigned char data_buffer_out_2_0[] = {0x06};
const unsigned char *const data_buffers_in[] = {data_buffer_in_0_0, data_buffer_in_1_0, data_buffer_in_2_0, };
const unsigned char *const data_buffers_out[] = {data_buffer_out_0_0, data_buffer_out_1_0, data_buffer_out_2_0, };
const size_t data_size_in[] = {sizeof(data_buffer_in_0_0), sizeof(data_buffer_in_1_0), sizeof(data_buffer_in_2_0), };
const size_t data_size_out[] = {sizeof(data_buffer_out_0_0), sizeof(data_buffer_out_1_0), sizeof(data_buffer_out_2_0), };
"""  # noqa: E501
    )


@pytest.mark.parametrize("files", [False, True])
@pytest.mark.parametrize("fmt", ["bin", "npy", "npz", "foo"])
def test_models_utils_lookup_data_buffers_legacy(tmp_path_factory, files, fmt):
    ins_dir = tmp_path_factory.mktemp("ins")
    outs_dir = tmp_path_factory.mktemp("outs")

    def helper(fname):
        if fmt in ["npy", "npz"]:
            compressed = fmt == "npz"
            if compressed:
                np.savez(fname, np.ones(1, dtype="int8"))
            else:
                np.save(fname, np.ones(1, dtype="int8"))
        else:  # bin, foo
            with open(fname, "wb") as f:
                f.write(bytes([1]))

    in_names = ["0", "1", "2"]
    out_names = ["0", "1", "2"]

    in_files = [ins_dir / f"{name}.{fmt}" for name in in_names]
    out_files = [outs_dir / f"{name}.{fmt}" for name in out_names]

    list(map(helper, in_files))
    list(map(helper, out_files))

    if files:
        ins, outs = lookup_data_buffers(in_files, out_files)
    else:
        ins, outs = lookup_data_buffers([ins_dir], [outs_dir])
    assert len(ins) == len(outs)

    if fmt in ["bin", "npy", "npz"]:
        assert ins == [["0x01, "], ["0x01, "], ["0x01, "]]
        assert outs == [["0x01, "], ["0x01, "], ["0x01, "]]
    else:
        assert ins == []
        assert outs == []


def test_models_utils_lookup_data_buffers_legacy_multi(tmp_path_factory):
    ins_dir = tmp_path_factory.mktemp("ins")
    outs_dir = tmp_path_factory.mktemp("outs")

    def helper(fname):
        with open(fname, "wb") as f:
            f.write(bytes([1]))

    in_names = ["0_0", "0_1", "1_0", "1_1", "2_0", "2_1"]
    out_names = ["0_0", "0_1", "0_2", "1_0", "1_1", "1_2", "2_0", "2_1", "2_2"]

    in_files = [ins_dir / f"{name}.bin" for name in in_names]
    out_files = [outs_dir / f"{name}.bin" for name in out_names]

    list(map(helper, in_files))
    list(map(helper, out_files))

    ins, outs = lookup_data_buffers([ins_dir], [outs_dir])
    assert len(ins) == len(outs)

    assert ins == [["0x01, ", "0x01, "], ["0x01, ", "0x01, "], ["0x01, ", "0x01, "]]
    assert outs == [["0x01, ", "0x01, ", "0x01, "], ["0x01, ", "0x01, ", "0x01, "], ["0x01, ", "0x01, ", "0x01, "]]


# def test_models_utils_lookup_data_buffers(tmp_path_factory):
#     data_dir = tmp_path_factory.mktemp("data")
# def test_models_utils_lookup_data_buffers_multi(tmp_path_factory):
#     data_dir = tmp_path_factory.mktemp("data")


def test_models_utils_get_data_source():
    # empty
    out = get_data_source([], [])
    assert (
        out
        == """#include "ml_interface.h"
#include <stddef.h>
const int num_data_buffers_in = 0;
const int num_data_buffers_out = 0;
const unsigned char *const data_buffers_in[] = {};
const unsigned char *const data_buffers_out[] = {};
const size_t data_size_in[] = {};
const size_t data_size_out[] = {};
"""
    )

    # non empty
    # too complex
