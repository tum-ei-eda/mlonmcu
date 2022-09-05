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
"""MLonMCU RISC-V utilities"""


def sort_extensions_canonical(extensions, lower=False, unpack=False):
    """Utility to get the canonical architecture name string."""

    # See: https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf#table.22.1
    ORDER = [
        "I",
        "M",
        "A",
        "F",
        "D",
        "G",
        "Q",
        "L",
        "C",
        "B",
        "J",
        "T",
        "P",
        "V",
        "X",
        "S",
        "SX",
    ]  # What about Z* extensions?
    extensions_new = extensions.copy()

    # make upper
    extensions_new = [x.upper() for x in extensions_new]

    if unpack:
        # Convert G into IMAFD
        if "G" in extensions_new:
            extensions_new = [x for x in extensions_new if x != "G"] + ["I", "M", "A", "F", "D"]
        # Remove duplicates
        extensions_new = list(set(extensions_new))

    def _get_index(x):
        if x in ORDER:
            return ORDER.index(x)
        else:
            for i, o in enumerate(ORDER):
                if x.startswith(o):
                    return i
            return ORDER.index("X") - 0.5  # Insert unknown keys right before custom extensions

    extensions_new.sort(key=lambda x: _get_index(x))

    if lower:
        extensions_new = [x.lower() for x in extensions_new]
    return extensions_new


def join_extensions(exts):
    sep = ""
    ret = ""
    for ext in exts:
        length = len(ext)
        if sep == "_":
            assert length > 1, "default extensions should come before any custom or sub-extensions"
        if length > 1:
            sep = "_"
        if ret != "":
            ret += sep
        ret += ext
    return ret
