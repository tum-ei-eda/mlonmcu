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
from io import StringIO

from mlonmcu.utils import is_power_of_two, ask_user, get_base_prefix_compat, in_virtualenv


def test_utils_is_power_of_two():
    assert not is_power_of_two(0)
    assert is_power_of_two(1)
    assert is_power_of_two(2)
    assert not is_power_of_two(3)
    assert is_power_of_two(4)
    assert is_power_of_two(8)
    assert is_power_of_two(16)
    assert not is_power_of_two(99)


def test_utils_ask_user_noninteractive():
    assert not ask_user("foo", False, interactive=False)
    assert ask_user("foo", True, interactive=False)


@pytest.mark.parametrize("default", [False, True])
def test_utils_ask_user_interactive(monkeypatch, default):
    monkeypatch.setattr("sys.stdin", StringIO("y\n"))
    assert ask_user("foo", default, interactive=True)  # y
    monkeypatch.setattr("sys.stdin", StringIO("j\n"))
    assert ask_user("foo", default, interactive=True)  # j
    monkeypatch.setattr("sys.stdin", StringIO("n\n"))
    assert not ask_user("foo", default, interactive=True)  # n
    monkeypatch.setattr("sys.stdin", StringIO("Y\n"))
    assert ask_user("foo", default, interactive=True)  # Y
    monkeypatch.setattr("sys.stdin", StringIO("J\n"))
    assert ask_user("foo", default, interactive=True)  # J
    monkeypatch.setattr("sys.stdin", StringIO("N\n"))
    assert not ask_user("foo", default, interactive=True)  # N
    monkeypatch.setattr("sys.stdin", StringIO("k\n"))
    assert ask_user("foo", default, interactive=True) is default  # k
    monkeypatch.setattr("sys.stdin", StringIO("K\n"))
    assert ask_user("foo", default, interactive=True) is default  # K
    monkeypatch.setattr("sys.stdin", StringIO("\n"))
    assert ask_user("foo", default, interactive=True) is default  # empty


def test_utils_get_base_prefix_compat():
    assert get_base_prefix_compat() is not None


def test_utils_in_virtualenv():
    assert isinstance(in_virtualenv(), bool)
