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
from argparse import Namespace
from mlonmcu.cli.helper.parse import parse_var, parse_vars, extract_feature_names, extract_config


def test_parse_var():
    assert parse_var("foo=bar") == ("foo", "bar")
    assert parse_var("foo=bar=2000") == ("foo", "bar=2000")
    # assert parse_var("foo=") == ("foo", None)


def test_parse_var_invalid():
    with pytest.raises(RuntimeError):
        parse_var("foo")
    with pytest.raises(RuntimeError):
        parse_var("=bar")


def test_parse_vars():
    assert parse_vars([]) == {}
    assert parse_vars([""]) == {}
    assert parse_vars(["foo=bar"]) == {"foo": "bar"}
    assert parse_vars(["foo=bar", "a=b"]) == {"foo": "bar", "a": "b"}


def test_extract_feature_names():
    assert extract_feature_names(Namespace(feature=None, feature_gen=None))[0] == []
    assert extract_feature_names(Namespace(feature=[], feature_gen=None))[0] == []
    assert extract_feature_names(Namespace(feature=["x"], feature_gen=None))[0] == ["x"]
    assert extract_feature_names(Namespace(feature=["x", "y"], feature_gen=None))[0] == ["x", "y"]


def test_extract_config():
    assert extract_config(Namespace(config=None, config_gen=None))[0] == {}
    assert extract_config(Namespace(config=[], config_gen=None))[0] == {}
    assert extract_config(Namespace(config=[[]], config_gen=None))[0] == {}
    assert extract_config(Namespace(config=[[], []], config_gen=None))[0] == {}
    assert extract_config(Namespace(config=[["foo=bar"]], config_gen=None))[0] == {"foo": "bar"}
    assert extract_config(Namespace(config=[["foo=bar", "a=b"]], config_gen=None))[0] == {"foo": "bar", "a": "b"}
    assert extract_config(Namespace(config=[["foo=bar"], ["a=b"]], config_gen=None))[0] == {"foo": "bar", "a": "b"}
