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
import re


def split_extensions(inp):
    inp = inp[4:]
    # special case for non std conform zve32x/zve64x
    matches = re.compile(r"(?:zve(?:32|64)x)|(?:[^xz_])|(?:x[^xz_]+)|(?:z[^xz_]+)").findall(inp)
    return set(matches)


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
        "Z",
        "XCVMAC",
        "XCVMEM",
        "XCVBI",
        "XCVALU",
        "XCVBITMANIP",
        "XCVSIMD",
        "XCVHWLP",
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


def join_extensions(exts, merge=True):
    sep = ""
    ret = ""
    if merge:
        if "i" in exts and "m" in exts and "a" in exts and "f" in exts and "d" in exts:
            exts = ["g"] + [e for e in exts if e not in "imafd"]
    for ext in exts:
        length = len(ext)
        if sep == "_":
            assert length > 1, "default extensions should come before any custom or sub-extensions"
        if length > 1 and ext not in ["xpulpv2", "xpulpv3", "xcorev"]:
            sep = "_"
        if ret != "":
            ret += sep
        ret += ext
    return ret


def update_extensions(
    exts,
    embedded=None,
    compressed=None,
    atomic=None,
    multiply=None,
    pext=None,
    pext_spec=None,
    vext=None,
    elen=None,
    embedded_vext=None,
    vlen=None,
    fpu=None,
    minimal=True,
    bext=None,
    bext_spec=None,
    bext_zba=None,
    bext_zbb=None,
    bext_zbc=None,
    bext_zbs=None,
):
    # ret = exts.copy()
    require = set()
    ignore_exts = ["zifencei", "zicsr"]
    for ext in exts:
        if ext == "g":
            fpu = "double"
            atomic = True
            multiply = True
        elif embedded is None and ext == "e":
            embedded = True
        elif multiply is None and ext == "m":
            multiply = True
        elif atomic is None and ext == "a":
            atomic = True
        elif compressed is None and ext == "c":
            compressed = True
        elif fpu is None and ext == "f":
            fpu = "single"
        elif (fpu is None or fpu == "single") and ext == "d":
            fpu = "double"
        elif vext is None and ext == "v":
            vext = True
        elif pext is None and ext == "p":
            pext = True
        elif vlen is None and "zvl" in ext and "sseg" not in ext:
            vlen_ = int(ext[3:-1])
            if vlen is None or vlen_ > vlen:
                vlen = vlen_
        elif embedded_vext is None and "zve" in ext:
            vext = True
            embedded_vext = True
            elen_ = int(ext[3:-1])
            if elen is None or elen_ > elen:
                elen = elen_
        # elif bext is None and ext in ["zba", "zbb", "zbc", "zbs"]:
        #     bext = True
        elif bext_zba is None and ext == "zba":
            bext_zba = True
        elif bext_zbb is None and ext == "zbb":
            bext_zbb = True
        elif bext_zbc is None and ext == "zbc":
            bext_zbc = True
        elif bext_zbs is None and ext == "zbs":
            bext_zbs = True
        elif ext in ignore_exts:
            pass
        else:
            require.add(ext)
    if embedded:
        require.add("e")
    else:
        require.add("i")
    if atomic:
        require.add("a")
    if multiply:
        require.add("m")
    if compressed:
        require.add("c")
    if fpu == "single":
        require.add("f")
    elif fpu == "double":
        require.add("d")
        require.add("f")
    if pext:
        require.add("p")
    if bext_zba:
        require.add("zba")
    if bext_zbb:
        require.add("zbb")
    if bext_zbc:
        require.add("zbc")
    if bext_zbs:
        require.add("zbs")
    if vext:
        if elen is None:
            elen = 32
        assert elen in [32, 64], f"Unsupported ELEN: {elen}"
        if elen == 32:  # Required to tell the compiler that EEW=64 is not allowed...
            if embedded_vext:
                if fpu in ["double", "single"]:
                    require.add("zve32f")
                else:
                    require.add("zve32x")
            else:
                assert fpu == "double"
                require.add("v")
        elif elen == 64:
            if embedded_vext:
                if fpu == "double":
                    require.add("zve64d")
                elif fpu == "single":
                    require.add("zve64f")
                else:
                    require.add("zve64x")
            else:
                assert fpu == "double"
                require.add("v")
        # if vlen:
        #     require.add(f"zvl{vlen}b")

    if not minimal:
        if fpu in ["single", "double"] and not minimal:
            require.add("zicsr")
        if vext or embedded_vext:
            require.add("zicsr")
        if atomic and multiply and fpu == "double":
            require.add("zifencei")

    ret = set()
    for ext in require:
        if ext not in ret:
            ret.add(ext)
    return ret


def update_extensions_pulp(exts, xpulp_version):
    ret = exts.copy()
    required = []
    if xpulp_version:
        required.append(f"xpulpv{xpulp_version}")
    for ext in required:
        if ext not in ret:
            ret.append(ext)
    return set(ret)
