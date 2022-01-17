import pytest
from argparse import Namespace
from mlonmcu.cli.helper.parse import parse_var, parse_vars, extract_feature_names, extract_config

def test_parse_var():
    assert parse_var("foo=bar") == ("foo", "bar")
    assert parse_var("foo=bar=2000") == ("foo", "bar=2000")
    # assert parse_var("foo=") == ("foo", None)

def test_parse_var_invalid():
    with pytest.raises(AssertionError):
        parse_var("foo")
    with pytest.raises(AssertionError):
        parse_var("=bar")

def test_parse_vars():
    assert parse_vars([]) == {}
    assert parse_vars([""]) == {}
    assert parse_vars(["foo=bar"]) == {"foo": "bar"}
    assert parse_vars(["foo=bar", "a=b"]) == {"foo": "bar", "a": "b"}

def test_extract_feature_names():
    assert extract_feature_names(Namespace(feature=None)) == []
    assert extract_feature_names(Namespace(feature=[])) == []
    assert extract_feature_names(Namespace(feature=["x"])) == ["x"]
    assert extract_feature_names(Namespace(feature=["x", "y"])) == ["x", "y"]

def test_extract_config():
    assert extract_config(Namespace(config=None)) == {}
    assert extract_config(Namespace(config=[])) == {}
    assert extract_config(Namespace(config=[[]])) == {}
    assert extract_config(Namespace(config=[[],[]])) == {}
    assert extract_config(Namespace(config=[["foo=bar"]])) == {"foo": "bar"}
    assert extract_config(Namespace(config=[["foo=bar", "a=b"]])) == {"foo": "bar", "a": "b"}
    assert extract_config(Namespace(config=[["foo=bar"], ["a=b"]])) == {"foo": "bar", "a": "b"}
