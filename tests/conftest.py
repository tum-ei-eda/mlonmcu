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

pytest_plugins = [
    "tests.fixtures",
]


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--run-user-context", action="store_true", default=False, help="run tests using user context")
    parser.addoption("--run-hardware", action="store_true", default=False, help="run tests using real hardware")


# def pytest_configure(config):
#     config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    slow = False
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        slow = True
    user_context = False
    if config.getoption("--run-user-context"):
        # --run-context given in cli: do not skip user context tests
        user_context = True
    hardware = False
    if config.getoption("--run-hardware"):
        # --run-hardware given in cli: do not skip hardware tests
        hardware = True
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_user_context = pytest.mark.skip(reason="need --run-context option to run")
    skip_hardware = pytest.mark.skip(reason="need --run-hardware option to run")
    for item in items:
        if not slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
        if not user_context and "user_context" in item.keywords:
            item.add_marker(skip_user_context)
        if not hardware and "hardware" in item.keywords:
            item.add_marker(skip_hardware)
