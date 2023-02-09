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
"""Test target registry mechanism."""
import pytest

from mlonmcu.target import Target, register_target, get_targets


class MyTarget(Target):
    def __init__(self, name="foo", features=None, config=None):
        super().__init__(name, features=features, config=config)


def test_target_registry():
    before = len(get_targets())
    register_target("foo", MyTarget)
    register_target("bar", MyTarget)
    assert len(get_targets()) == before + 2
    with pytest.raises(RuntimeError):
        register_target("foo", MyTarget)
    assert len(get_targets()) == before + 2
    register_target("foo", MyTarget, override=True)
    assert len(get_targets()) == before + 2
