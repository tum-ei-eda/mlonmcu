#!/usr/bin/env python
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

"""Tests for `mlonmcu` package."""


# import unittest
import pytest

# import os
# import tempfile
import configparser
from pathlib import Path

# from pytest_console_scripts import ScriptRunner
# import xdg
# import mock

from mlonmcu.version import __version__
from mlonmcu.cli.main import main


# @pytest.fixture()
# def fake_config_home(tmp_path):
#     config_home = tmp_path / "cfg"
#     config_home.mkdir()
#     mlonmcu_config_home = config_home / "mlonmcu"
#     mlonmcu_config_home.mkdir()
#     patcher = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_home)})
#     patcher.start()
#     yield mlonmcu_config_home
#     patcher.stop()
#
#
# @pytest.fixture()
# def fake_environment_directory(tmp_path):
#     cwd = tmp_path / "home"
#     cwd.mkdir()
#     yield cwd
#
#
# @pytest.fixture()
# def fake_working_directory(tmp_path):
#     cwd = tmp_path / "cwd"
#     cwd.mkdir()
#     yield cwd


@pytest.mark.script_launch_mode("subprocess")
def test_version(capsys):
    with pytest.raises(SystemExit) as pytest_exit:
        main(["--version"])
    out, err = capsys.readouterr()
    assert pytest_exit.value.code == 0
    assert out == f"mlonmcu {__version__}\n"
    assert err == ""


def _write_environments_file(path, data):
    config = configparser.ConfigParser()
    for key in data:
        config[key] = data[key]
    with open(path, "w") as env_file:
        config.write(env_file)
    print("_write_environments_file", path, data)


def _create_empty_environments_file(path):
    _write_environments_file(path, {})
    # with open(path, "w") as env_file:
    #     env_file.write("")


def _create_valid_environments_file(path):
    _write_environments_file(
        path,
        {
            "default": {"path": "/x/y/z", "created": "20211223T08000"},
            "custom": {"path": "/a/b/c"},
        },
    )
    # with open(path, "w") as env_file:
    #     env_file.write("""[default]\npath=\"/x/y/z\"\ncreated=\"20211223T08000\"\n\n[custom]\npath=\"/a/b/c\"""")


def _create_complex_environments_file(path):
    _write_environments_file(
        path,
        {
            "default": {"path": "/x/y/z", "created": "20211223T08000"},
            "custom": {"path": "/a/b/c"},
        },
    )
    # with open(path, "w") as env_file:
    #     env_file.write("""[default]\npath=\"/x/y/z\"\ncreated=\"20211223T08000\"\n\n[custom]\npath=\"/a/b/c\"""")


def _create_invalid_environments_file(path):
    _write_environments_file(path, {"default": {}})
    # with open(path, "w") as env_file:
    #     env_file.write("[default]")


def _count_envs(text):
    count = 0
    for line in text.split("\n"):
        if "    - " in line:
            count = count + 1
    return count


@pytest.mark.script_launch_mode("subprocess")
def test_env_empty(fake_config_home: Path, capsys):
    _create_empty_environments_file(fake_config_home / "environments.ini")
    ret = main(["env"])
    out, err = capsys.readouterr()
    assert ret == 0
    count = _count_envs(out)
    assert count == 0


# @pytest.mark.script_launch_mode("subprocess")
def test_env_valid(fake_config_home: Path, capsys):
    _create_valid_environments_file(fake_config_home / "environments.ini")
    # ret = script_runner.run("mlonmcu", "env")
    ret = main(["env"])
    out, err = capsys.readouterr()
    assert ret == 0
    count = _count_envs(out)
    assert count > 0


# @pytest.mark.script_launch_mode('subprocess')
# def test_env_file(script_runner: ScriptRunner, fake_config_home: Path):
#     ret = script_runner.run('mlonmcu', 'env', '--file')
#     assert ret.success
#     assert dirname in ret.stdout
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_env_add(script_runner: ScriptRunner, fake_config_home: Path):
#     _create_valid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'env', 'add', 'new', '/foo/bar')
#     assert ret.success
#     # TODO
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_env_add_invalid(script_runner: ScriptRunner, fake_config_home: Path):  # already exists
#     _create_valid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'env', 'add', 'default', '/foo/bar')
#     assert ret.success
#     # TODO
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_env_delete(script_runner: ScriptRunner, fake_config_home: Path):
#     _create_valid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'env', 'delete', 'custom')
#     assert ret.success
#     # TODO
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_env_delete_invalid(script_runner: ScriptRunner, fake_config_home: Path):  # does not exits
#     _create_valid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'env', 'delete', 'custom2')
#     assert ret.success
#     # TODO
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_env_delete_invalid(script_runner: ScriptRunner, fake_config_home: Path):  # does not exits
#     _create_valid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'env', 'delete', 'custom2')
#     assert ret.success
#     # TODO
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_env_update(script_runner: ScriptRunner, fake_config_home: Path):
#     _create_valid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'env', 'update', 'custom', '/foo/bar')
#     assert ret.success
#     # TODO
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_env_update_invalid(script_runner: ScriptRunner, fake_config_home: Path):  # does not exist
#     _create_valid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'env', 'update', 'custom2', '/foo/bar')
#     assert ret.success
#     # TODO
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_env_invalid2(script_runner: ScriptRunner, fake_config_home: Path):
#     _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'env')
#     assert not ret.success

# class MockedEnvironmentDirectory:
#     def __init__(self):
#         self.tempdir = None
#         self.env_dir = None
#         self.env_file_path = None
#     def __enter__(self):
#         self.tempdir = tempfile.TemporaryDirectory()
#         self.env_dir = Path(self.tempdir.name) / "mlonmcu"
#         os.mkdir(self.env_dir)
#         self.env_file_path = self.env_dir / "environments.ini"
#         return self
#     def __exit__(self, type, value, traceback):
#         self.tempdir.cleanup()
#         self.env_dir = None
#         self.env_file_path = None
#
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_env_invalid_compact(script_runner):
#     with MockedEnvironmentDirectory() as mocked:
#         _create_invalid_environments_file(mocked.env_file_path)
#         ret = script_runner.run('mlonmcu', 'env')
#         assert not ret.success

# TODO:
# test_cleanup_all
# test_cleanup_deps
# test_cleanup_results
# test_cleanup_sessions
# test_cleanup_session

# @pytest.mark.script_launch_mode('subprocess')
# def test_cleanup_all(script_runner):
#     with tempfile.TemporaryDirectory() as dirname:
#         env_dir = Path(dirname) / "mlonmcu"
#         os.mkdir(env_dir)
#         env_file_path = env_dir / "environments.ini"
#         _create_valid_environments_file(env_file_path)
#         with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": dirname}):
#             ret = script_runner.run('mlonmcu', 'env')
#             assert ret.success
#             count = _count_envs(ret.stdout)
#             assert count > 0

# @pytest.mark.script_launch_mode('subprocess')
# def test_init_local(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init', str(fake_environment_directory))
#     # ensure initialized
#     # ensure not registered
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_init_local_empty(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init', str(fake_environment_directory))
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_init_local_nonempty(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init', str(fake_environment_directory))
#     assert not ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_init_local_named(script_runner: ScriptRunner, fake_config_home: Path, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init', str(fake_environment_directory), '--name', 'local')
#     # ensure initialized
#     # ensure registered
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_init_local_named_duplicate(script_runner: ScriptRunner, fake_config_home: Path,
#         fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init', str(fake_environment_directory), '--name', 'local')
#     # ensure initialized
#     # ensure registered
#     assert not ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_init_default(script_runner: ScriptRunner, fake_config_home: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_init_default_directory_exists(script_runner: ScriptRunner, fake_config_home: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init')
#     assert not ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_init_default_name_exists(script_runner: ScriptRunner, fake_config_home: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init')
#     assert not ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_init_managed(script_runner: ScriptRunner, fake_config_home: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init', '--name', 'my_env')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_init_managed_invalid_name(script_runner: ScriptRunner, fake_config_home: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'init' '--name', 'foo bar')
#     assert not ret.success
#
# # TODO: init cmdline flags for create venv, run setup?, template, list templates
# # mlonmcu validate? (env directories and environment file + user config consistency)
#
# NOP? -> mock task/dependency stuff with a few test tasks
# @pytest.mark.script_launch_mode('subprocess')
# def test_setup_default(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'setup')
#     assert ret.success
#
# NOP? -> mock task/dependency stuff with a few test tasks
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_default(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_by_name(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', '--home', 'my_env')
#     assert ret.success
#
# # TODO: --home -H -> --env -E?
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_by_path_directory(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', '--home', str(fake_environment_directory))
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_by_path_file(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', '--home', str(fake_environment_directory))
#     assert ret.success
#
# # TODO: PATH > NAME > CWD > ENVVAR > DEFAULT
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_by_env_var(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_by_cwd(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_load(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', 'load')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_load(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', 'build')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_load(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', 'compile')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_load(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', 'run')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_test(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', 'test')
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_test_batch(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', 'test', 'modelA', 'modelB', '--backend' 'backendA', '--backend',
#         'backendB', '--target', 'targetA', '--target', 'targetB')
#     # TODO: create helper function for creating test batch or single
#     assert ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_continue(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', 'load', ...)
#     assert ret.success
#     ret = script_runner.run('mlonmcu', 'flow', 'build', '--continue')
#     assert ret.success
#     # assert same session
#     ret = script_runner.run('mlonmcu', 'flow', 'compile', '--continue')
#     assert ret.success
#     # assert same session
#     ret = script_runner.run('mlonmcu', 'flow', 'run', '--continue')
#     assert ret.success
#     # assert same session
#     ret = script_runner.run('mlonmcu', 'flow', 'test', '--continue')
#     assert ret.success
#     # assert same session
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_continue_incompatible(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', 'build', ...)
#     assert ret.success
#     # assert same session
#     ret = script_runner.run('mlonmcu', 'flow', 'compile', '--continue', '--feature', 'featureA')
#     assert not ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_flow_continue_missing_parent(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'flow', 'compile', '--continue')
#     assert not ret.success
#
# @pytest.mark.script_launch_mode('subprocess')
# def test_export(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'export')
#     assert ret.success
# mlonmcu export -> Cleanup?
# test_export_list
# test_export_session_list
# test_export_run_list
# test_export_session
# test_export_run
# test_export_invalid_session
# test_export_invalid_run
# test_export_zip
# test_export_dir


# @pytest.mark.script_launch_mode('subprocess')
# def test_models(script_runner: ScriptRunner, fake_environment_directory: Path):
#     # _create_invalid_environments_file(fake_config_home / "environments.ini")
#     ret = script_runner.run('mlonmcu', 'models')
#     assert ret.success
