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

# init
import os
import mock
import pytest
import configparser
from mlonmcu.environment.init import clone_models_repo, create_environment_directories, create_venv_directory, ask_user, initialize_environment
from mlonmcu.environment.list import get_alternative_name
from mlonmcu.environment.templates import fill_template, write_environment_yaml_from_template

def test_environment_create_environment_directories(fake_environment_directory):
    directories = ["foo", "bar", "test"]
    create_environment_directories(fake_environment_directory, directories)
    assert all([(fake_environment_directory / d).is_dir() for d in directories])

# def test_environment_clone_models_repo(fake_environment_directory):
#     subdir = fake_environment_directory / "models"
#     clone_models_repo(subdir)
#     assert (subdir / ".git").is_dir()


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("default", [False, True])
def test_environment_ask_user(interactive, default):
    msg = "Answer?"
    # empty or implicit
    with mock.patch('builtins.input', return_value=''):
        assert ask_user(msg, default=default, interactive=interactive) == default
    with mock.patch('builtins.input', return_value='foo'):
        assert ask_user(msg, default=default, interactive=interactive) == default
    # yes (explicit)
    with mock.patch('builtins.input', return_value='y'):
        if interactive:
            assert ask_user(msg, default=default, interactive=interactive)
        else:
            assert ask_user(msg, default=default, interactive=interactive) == default
    with mock.patch('builtins.input', return_value='Y'):
        if interactive:
            assert ask_user(msg, default=default, interactive=interactive)
        else:
            assert ask_user(msg, default=default, interactive=interactive) == default
    # no  (explicit)
    with mock.patch('builtins.input', return_value='n'):
        if interactive:
            assert not ask_user(msg, default=default, interactive=interactive)
        else:
            assert ask_user(msg, default=default, interactive=interactive) == default
    with mock.patch('builtins.input', return_value='N'):
        if interactive:
            assert not ask_user(msg, default=default, interactive=interactive)
        else:
            assert ask_user(msg, default=default, interactive=interactive) == default

@pytest.mark.parametrize("hidden", [False, True])
def test_environment_create_venv_directory(hidden, fake_environment_directory):
    create_venv_directory(fake_environment_directory, hidden=hidden)
    assert (fake_environment_directory / (".venv" if hidden else "venv")).is_dir()

def _count_envs(directory):
    envs_file = directory / "environments.ini"
    parser = configparser.ConfigParser()
    parser.read(str(envs_file))
    return len(parser.sections())

def _has_duplicate_envs(directory):
    envs_file = directory / "environments.ini"
    parser = configparser.ConfigParser()
    parser.read(str(envs_file))
    secs = parser.sections()
    unique = list(dict.fromkeys(secs))  # Without duplicates
    return len(unique) != len(secs)

def test_environment_get_environment_names():
    pass  # TODO

def test_environment_get_alternative_name():
    assert get_alternative_name("foo", ["foo", "bar", "foo_bar"]) == "foo_0"
    assert get_alternative_name("foo", ["foo", "bar", "foo_bar", "foo_0"]) == "foo_1"
    assert get_alternative_name("foo_9", ["foo_9", "foo_10", "foo_9_0"]) == "foo_9_1"

def test_environment_register_environment():
    pass  # TODO

@pytest.mark.parametrize("interactive", [False])  # We do not want to mock the user input here
@pytest.mark.parametrize("create_venv", [False])  # Already covered above
@pytest.mark.parametrize("clone_models", [False])  # Already covered above
@pytest.mark.parametrize("skip_sw", [True])  # Already covered above
@pytest.mark.parametrize("register", [False, True])  #
@pytest.mark.parametrize("template", ["default"])
def test_environment_initialize_environment(interactive, create_venv, clone_models, skip_sw, register, template, fake_environment_directory, fake_working_directory, fake_config_home):
    with mock.patch("mlonmcu.environment.templates.get_template_text", return_value=str.encode("---")) as mocked:
        before = _count_envs(fake_config_home)
        initialize_environment(fake_environment_directory, "myenv", interactive=interactive, create_venv=create_venv, clone_models=clone_models, skip_sw=skip_sw, register=register, template=template)
        initialize_environment(fake_working_directory, "another", interactive=interactive, create_venv=create_venv, clone_models=clone_models, skip_sw=skip_sw, register=register, template=template)
        assert _count_envs(fake_config_home) == ((before + 2) if register else before)

def test_environment_initialize_environment_duplicate(fake_environment_directory, fake_working_directory, fake_config_home):
    with mock.patch("mlonmcu.environment.templates.get_template_text", return_value=str.encode("---")) as mocked:
        before = _count_envs(fake_config_home)
        initialize_environment(fake_environment_directory, "", interactive=False, create_venv=False, clone_models=False, skip_sw=True, register=True, template="default")
        initialize_environment(fake_working_directory, "", interactive=False, create_venv=False, clone_models=False, skip_sw=True, register=True, template="default")
        assert _count_envs(fake_config_home) == (before + 2)
        assert not _has_duplicate_envs(fake_config_home)

# template

def test_environment_fill_template():
    with mock.patch("mlonmcu.environment.templates.get_template_text", return_value=str.encode("{{ key }}: {{ value }}")) as mocked:
        assert fill_template("", data={"key": "foo", "value": "bar"}) == "foo: bar"
    with mock.patch("mlonmcu.environment.templates.get_template_text", return_value=None) as mocked:
        assert fill_template("", data={"key": "foo", "value": "bar"}) == None

def test_environment_write_environment_yaml_from_template(fake_environment_directory, fake_config_home):
    with mock.patch("mlonmcu.environment.templates.get_template_text", return_value=str.encode("home: {{ home_dir }}\nconfig: {{ config_dir }}")) as mocked:
        env_yaml = str(fake_environment_directory / "environment.yml")
        write_environment_yaml_from_template(env_yaml, "", fake_environment_directory)
        assert os.path.isfile(env_yaml)
        with open(env_yaml) as yaml:
            lines = "".join(yaml.readlines())
            assert lines == f"home: {fake_environment_directory}\nconfig: {fake_config_home}"
