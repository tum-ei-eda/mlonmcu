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

from mlonmcu.target.riscv.util import (
    sort_extensions_canonical,
    join_extensions,
    update_extensions,
    update_extensions_pulp,
)


def test_target_riscv_sort_extensions_canonical():
    # default
    assert sort_extensions_canonical(["c", "g"]) == ["G", "C"]
    assert sort_extensions_canonical(["i", "m", "a", "c", "f", "d"]) == ["I", "M", "A", "F", "D", "C"]
    assert sort_extensions_canonical(["g", "zifencei", "xcustom", "c"]) == ["G", "C", "ZIFENCEI", "XCUSTOM"]

    # lower
    assert sort_extensions_canonical(["c", "g"], lower=True) == ["g", "c"]

    # unpack
    assert sort_extensions_canonical(["c", "g"], unpack=True) == ["I", "M", "A", "F", "D", "C"]

    # lower + unpack
    assert sort_extensions_canonical(["c", "g"], lower=True, unpack=True) == ["i", "m", "a", "f", "d", "c"]

    # TODO: mixed cases


def test_target_riscv_join_extensions():
    assert join_extensions(["g", "c"]) == "gc"
    assert join_extensions(["g", "c", "zifencei"]) == "gc_zifencei"
    assert join_extensions(["g", "c", "xcustom"]) == "gc_xcustom"
    assert join_extensions(["g", "c", "xcorev"]) == "gcxcorev"

    with pytest.raises(AssertionError):
        join_extensions(["g", "xcustom", "c"])


def test_target_riscv_update_extensions():
    # default
    assert update_extensions(["i", "m", "c"]) == {"i", "m", "c"}

    # pext
    assert update_extensions(["i", "m", "c"], pext=True) == {"i", "m", "c", "p"}
    assert update_extensions(["i", "m", "c", "p"], pext=True) == {"i", "m", "c", "p"}
    # assert update_extensions(["i", "m", "c"], pext=True, variant="xuantie", pext_spec=0.94) == {
    #     "i",
    #     "m",
    #     "c",
    #     "p",
    #     "zpn",
    #     "zpsfoperand",
    # }
    # assert update_extensions(["i", "m", "c"], pext=True, variant="xuantie", pext_spec=0.97) == {
    #     "i",
    #     "m",
    #     "c",
    #     "p",
    #     "zpn",
    #     "zpsfoperand",
    #     "zbpbo",
    # }
    # assert update_extensions(["i", "m", "c", "p"], pext=False) == {"i", "m", "c", "?"}

    # vext
    assert update_extensions(["g", "c"], vext=True, fpu="double", elen=32) == {"i", "m", "a", "f", "d", "c", "v"}
    assert update_extensions(["g", "c"], vext=True, fpu="double", elen=64) == {"i", "m", "a", "f", "d", "c", "v"}
    # with pytest.raises(AssertionError):
    #     assert update_extensions(["g", "c"], vext=True, fpu="single")
    # with pytest.raises(AssertionError):
    #     assert update_extensions(["g", "c"], vext=True, fpu=None)

    # embedded vext
    base = {"i", "m", "a", "c"}
    assert update_extensions(base, vext=True, fpu=None, embedded_vext=True, elen=32) == {"i", "m", "a", "c", "zve32x"}
    assert update_extensions(base, vext=True, fpu=None, embedded_vext=True, elen=64) == {"i", "m", "a", "c", "zve64x"}
    assert update_extensions(base, vext=True, fpu="single", embedded_vext=True, elen=32) == {
        "i",
        "m",
        "a",
        "f",
        "c",
        "zve32f",
    }
    assert update_extensions(base, vext=True, fpu="single", embedded_vext=True, elen=64) == {
        "i",
        "m",
        "a",
        "f",
        "c",
        # "zve32f",
        "zve64f",
    }
    assert update_extensions(base, vext=True, fpu="double", embedded_vext=True, elen=64) == {
        "i",
        "m",
        "a",
        "f",
        "d",
        "c",
        # "zve32f",
        "zve64d",
    }

    # TODO: fpu, multiple elen


def test_target_riscv_update_extensions_pulp():
    assert update_extensions_pulp(["i", "m", "c"], None) == {"i", "m", "c"}
    assert update_extensions_pulp(["i", "m", "c"], 2) == {"i", "m", "c", "xpulpv2"}
