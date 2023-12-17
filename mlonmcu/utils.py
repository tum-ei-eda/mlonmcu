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
import sys


def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0


def ask_user(text, default: bool, yes_keys=["y", "j"], no_keys=["n"], interactive=True):
    assert len(yes_keys) > 0 and len(no_keys) > 0
    if not interactive:
        return default
    if default:
        suffix = " [{}/{}] ".format(yes_keys[0].upper(), no_keys[0].lower())
    else:
        suffix = " [{}/{}] ".format(yes_keys[0].lower(), no_keys[0].upper())
    message = text + suffix
    answer = input(message)
    if default:
        return answer.lower() not in no_keys and answer.upper() not in no_keys
    else:
        return not (answer.lower() not in yes_keys and answer.upper() not in yes_keys)


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix


def in_virtualenv():
    """Detects if the current python interpreter is from a virtual environment."""
    return get_base_prefix_compat() != sys.prefix


def filter_none(data):
    """Helper function which drop dict items with a None value."""
    assert isinstance(data, dict), "Dict only"
    out = {key: value for key, value in data.items() if value is not None}
    return out
