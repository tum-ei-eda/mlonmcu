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

from mlonmcu.config import filter_config, remove_config_prefix, resolve_required_config, str2bool, str2dict


def test_remove_config_prefix_supports_exact_and_wildcard_prefixes():
    config = {"tvm.opt": 3, "tvmaot.opt": 2, "global": 1, "tvm.keep": 4}
    assert remove_config_prefix(config, "tvm", skip={"tvm.keep"}) == {"opt": 3}
    assert remove_config_prefix({"tvm*.opt": 2}, "tvmaot") == {"opt": 2}


def test_filter_config_merges_scoped_global_and_default_values():
    config = {"demo.local": 1, "required": 2, "optional": 3, "unrelated": 9}
    result = filter_config(config, "demo", {"default": 4}, {"optional", "missing"}, {"required"})
    assert result == {"local": 1, "required": 2, "optional": 3, "missing": None, "default": 4}

    with pytest.raises(AssertionError, match="Required config key"):
        filter_config({}, "demo", {}, set(), {"required"})


class FakeCache(dict):
    pass


def test_resolve_required_config_uses_config_cache_hints_and_optionals():
    cache = FakeCache({("tool", ("fast",)): "/bin/tool", ("optional", ()): "present"})
    result = resolve_required_config(
        {"tool", "configured"},
        optional={"optional", "absent"},
        config={"configured": "yes"},
        cache=cache,
        hints=["fast"],
    )
    assert result == {"tool": "/bin/tool", "configured": "yes", "optional": "present"}


def test_resolve_required_config_reports_missing_dependencies():
    with pytest.raises(AssertionError, match="No dependency cache"):
        resolve_required_config({"tool"})
    with pytest.raises(RuntimeError, match="empty"):
        resolve_required_config({"tool"}, cache={})
    with pytest.raises(RuntimeError, match="cache miss"):
        resolve_required_config({"tool"}, cache={("other", ()): "x"})


@pytest.mark.parametrize("value", [True, 1, "yes", "TRUE", "on", "1"])
def test_str2bool_true_values(value):
    assert str2bool(value) is True


@pytest.mark.parametrize("value", [False, 0, "no", "FALSE", "off", "0"])
def test_str2bool_false_values(value):
    assert str2bool(value) is False


def test_str_converters_handle_none_and_invalid_values():
    assert str2bool(None, allow_none=True) is None
    assert str2dict(None, allow_none=True) is None
    assert str2dict({"a": 1}) == {"a": 1}
    assert str2dict("{'a': 1}") == {"a": 1}
    with pytest.raises(ValueError):
        str2bool("perhaps")
    with pytest.raises((ValueError, SyntaxError)):
        str2dict("not a dictionary")
