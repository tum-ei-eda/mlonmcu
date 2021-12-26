#!/usr/bin/env python

"""Tests for `mlonmcu` package."""


# import unittest
import pytest
import os
import tempfile
import configparser
from pathlib import Path
import xdg
import mock

from mlonmcu.version import __version__
# from mlonmcu import mlonmcu
#
#
# class TestMlonmcuCli(unittest.TestCase):
#     """Tests for `mlonmcu` package."""
#
#     def setUp(self):
#         """Set up test fixtures, if any."""
#
#     def tearDown(self):
#         """Tear down test fixtures, if any."""
#
#     def test_version(self):
#         """Test printing of version number."""

@pytest.mark.script_launch_mode('subprocess')
def test_version(script_runner):
    ret = script_runner.run('mlonmcu', '--version')
    assert ret.success
    assert ret.stdout == f'mlonmcu {__version__}\n'
    assert ret.stderr == ''

def _write_environments_file(path, data):
    config = configparser.ConfigParser()
    for key in data:
        config[key] = data[key]
    with open(path, "w") as env_file:
        config.write(env_file)

def _create_empty_environments_file(path):
    _write_environments_file(path, {})

def _create_valid_environments_file(path):
    _write_environments_file(path, {"default": {"path": "/x/y/z", "created": "20211223T08000"}, "cutom": {"path": "/a/b/c"}})

def _create_invalid_environments_file(path):
    _write_environments_file(path, {"default": {}})

@pytest.mark.script_launch_mode('subprocess')
def test_env_empty(script_runner):
    with tempfile.TemporaryDirectory() as dirname:
        env_dir = Path(dirname) / "mlonmcu"
        os.mkdir(env_dir)
        env_file_path = env_dir / "environments.ini"
        _create_empty_environments_file(env_file_path)
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": dirname}):
            ret = script_runner.run('mlonmcu', 'env')
            assert ret.success
            # assert ret.stdout == f'mlonmcu {__version__}\n'
            # assert ret.stderr == ''

@pytest.mark.script_launch_mode('subprocess')
def test_env_valid(script_runner):
    with tempfile.TemporaryDirectory() as dirname:
        env_dir = Path(dirname) / "mlonmcu"
        os.mkdir(env_dir)
        env_file_path = env_dir / "environments.ini"
        _create_valid_environments_file(env_file_path)
        with mock.patch.dict(os.environ, {"xdg_config_home": dirname}):
            ret = script_runner.run('mlonmcu', 'env')
            assert ret.success
            # assert ret.stdout == f'mlonmcu {__version__}\n'
            # assert ret.stderr == ''

@pytest.mark.script_launch_mode('subprocess')
def test_env_invalid(script_runner):
    with tempfile.TemporaryDirectory() as dirname:
        env_dir = Path(dirname) / "mlonmcu"
        os.mkdir(env_dir)
        env_file_path = env_dir / "environments.ini"
        _create_invalid_environments_file(env_file_path)
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": dirname}):
            ret = script_runner.run('mlonmcu', 'env')
            assert not ret.success
            # assert ret.stdout == f'mlonmcu {__version__}\n'
            # assert ret.stderr == ''
