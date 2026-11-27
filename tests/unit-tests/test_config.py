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
from unittest.mock import patch

import pytest

from mlonmcu.config import (
    ConfigTracker,
    Configurable,
    ConfigField,
    cfg,
    filter_config,
    pick_first,
    remove_config_prefix,
    resolve_required_config,
    str2bool,
    str2dict,
    str2list,
    optional,
    parse_int,
    required,
)


def test_config_fields_metadata_access_and_legalization():
    class Base(Configurable):
        enabled = cfg(False, cast=str2bool)
        nullable = cfg(None, cast=lambda value: (_ for _ in ()).throw(RuntimeError("called")))
        source = required("external.source")
        tool = optional("external.tool")

    class Child(Base):
        enabled = required("child.enabled", cast=str2bool)
        port = cfg(12, cast=int)

    assert isinstance(Base.enabled, ConfigField)
    assert Base.DEFAULTS == {"enabled": False, "nullable": None}
    assert Base.REQUIRED == {"external.source"}
    assert Base.OPTIONAL == {"external.tool"}
    assert Child.DEFAULTS == {"nullable": None, "port": 12}
    assert Child.REQUIRED == {"external.source", "child.enabled"}
    assert Child.OPTIONAL == {"external.tool"}
    assert Base.DEFAULTS == {"enabled": False, "nullable": None}  # no parent mutation
    obj = Child.__new__(Child)
    obj.config = {
        "child.enabled": "yes",
        "external.source": "src",
        "external.tool": None,
        "nullable": None,
        "port": "4",
    }
    assert obj.enabled is True
    assert obj.nullable is None
    assert obj.source == "src"
    assert obj.tool is None
    assert obj.port == 4
    assert set(Child.CONFIG_FIELDS) == {"enabled", "nullable", "source", "tool", "port"}


def test_config_fields_compose_with_legacy_metadata():
    class Legacy(Configurable):
        DEFAULTS = {"old": 1, "reclassified": 2}
        REQUIRED = {"needed"}
        OPTIONAL = {"maybe"}

    class Modern(Legacy):
        reclassified = required(cast=str)
        maybe = cfg("now-default")

    assert Modern.DEFAULTS == {"old": 1, "maybe": "now-default"}
    assert Modern.REQUIRED == {"needed", "reclassified"}
    assert Modern.OPTIONAL == set()


@pytest.mark.parametrize("value, expected", [(4, 4), ("42", 42), ("0x20", 32)])
def test_parse_int(value, expected):
    assert parse_int(value) == expected


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


def test_config_tracker_tracks_used_keys_and_ignores_known_keys(capsys):
    tracker = ConfigTracker({"used": 1, "unused": 2, "runs_per_stage": 3})

    tracker.track("used")
    tracker.track_all({"another"})

    assert tracker.used == {"used", "another"}
    assert tracker.unused == {"unused"}
    assert tracker.unknown == {"another"}
    assert capsys.readouterr().out == ""


def test_config_tracker_check_warns_about_unused_keys(capsys):
    tracker = ConfigTracker({"used": 1, "unused": 2})
    tracker.track("used")

    with patch("mlonmcu.config.logger.warning") as warning:
        assert tracker.check() is True

    warning.assert_called_once_with("Unused config vars found: unused")
    assert capsys.readouterr().out == ""


def test_config_tracker_check_succeeds_when_all_keys_are_used():
    tracker = ConfigTracker({"used": 1})
    tracker.track("used")

    with patch("mlonmcu.config.logger.warning") as warning:
        assert tracker.check() is False

    warning.assert_not_called()


def test_config_tracker_check_fails_on_unused_keys():
    tracker = ConfigTracker({"unused": 1})

    with pytest.raises(RuntimeError, match="Unused config vars found: unused"):
        tracker.check(fail_on_err=True)


def test_config_tracker_merge_combines_names_and_used_keys(capsys):
    first = ConfigTracker({"first": 1})
    first.track("first")
    second = ConfigTracker({"second": 2})
    second.track("second")

    merged = first.merge(second)

    assert merged.names == {"first", "second"}
    assert merged.used == {"first", "second"}
    assert merged.unused == set()
    assert capsys.readouterr().out == ""


def test_filter_config_tracker_records_scoped_and_global_keys(capsys):
    tracker = ConfigTracker({"demo.enabled": True, "required": "value"})

    config = filter_config(
        {"demo.enabled": True, "required": "value"},
        "demo",
        {"enabled": False},
        set(),
        {"required"},
        config_tracker=tracker,
    )

    assert config == {"enabled": True, "required": "value"}
    assert tracker.used == {"demo.enabled", "required"}
    assert capsys.readouterr().out == ""


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


@pytest.mark.parametrize(
    "value,expected",
    [
        (["a", "b"], ["a", "b"]),
        (("a", "b"), ["a", "b"]),
        ({"a", "b"}, ["a", "b"]),
        ("a,b", ["a", "b"]),
        ("['a', 'b']", ["a", "b"]),
    ],
)
def test_str2list(value, expected):
    assert sorted(str2list(value)) == sorted(expected)


def test_str2list_none_and_invalid_type():
    assert str2list(None, allow_none=True) is None
    with pytest.raises(AssertionError):
        str2list(None)
    with pytest.raises(AssertionError):
        str2list(1.5)


def test_pick_first_selects_first_valid_candidate():
    config = {"first": None, "second": 2, "third": 3}
    assert pick_first(config, ["missing", "first", "second", "third"]) == 2
    assert pick_first(config, ["first"], allow_none=True) is None


def test_pick_first_empty_and_missing_behavior():
    assert pick_first({}, [], allow_empty=True) is None
    assert pick_first({}, ["missing"], allow_fail=True) is None
    with pytest.raises(AssertionError, match="empty list"):
        pick_first({}, [])
    with pytest.raises(AssertionError, match="could not find"):
        pick_first({}, ["missing"])
