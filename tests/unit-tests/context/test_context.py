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
from pathlib import Path
import pytest
from mlonmcu.context.context import MlonMcuContext


def create_minimal_environment_yaml(path):
    dirname = path.parent.absolute()
    with open(path, "w") as f:
        f.write(f"---\nhome: {dirname}")  # Use defaults


def create_invalid_environment_yaml(path):
    dirname = path.parent.absolute()
    with open(path, "w") as f:
        f.write(f"---\nhome: {dirname}")  # Use defaults


# def test_resolve_environment_file():
# def test_resolve_environment_file_by_name():
# def test_resolve_environment_file_by_file():
# def test_resolve_environment_file_by_dir():
# def test_resolve_environment_file_by_cwd():
# def test_resolve_environment_file_by_env():
# def test_resolve_environment_file_by_default():
# def test_load_recent_sessions():
# def test_create_session():
# def test_load_cache():
# def test_get_session():


def test_open_context(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
    monkeypatch.chdir(fake_environment_directory)
    create_minimal_environment_yaml(fake_environment_directory / "environment.yml")
    ctx = None
    with MlonMcuContext() as context:
        assert context
        ctx = context
    assert ctx.is_clean


# def test_open_context_by_env(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
#     monkeypatch.chdir(fake_environment_directory)
#     create_minimal_environment_yaml(fake_environment_directory / "environment.yml")
#     with mlonmcu.context.MlonMcuContext() as context:
#         assert context
#
# def test_open_context_by_default(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
#     monkeypatch.chdir(fake_environment_directory)
#     create_minimal_environment_yaml(fake_environment_directory / "environment.yml")
#     with mlonmcu.context.MlonMcuContext() as context:
#         assert context
#
# def test_open_context_by_path(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
#     monkeypatch.chdir(fake_environment_directory)
#     create_minimal_environment_yaml(fake_environment_directory / "environment.yml")
#     with mlonmcu.context.MlonMcuContext() as context:
#         assert context
#
# def test_open_context_by_name(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
#     monkeypatch.chdir(fake_environment_directory)
#     create_minimal_environment_yaml(fake_environment_directory / "environment.yml")
#     with mlonmcu.context.MlonMcuContext() as context:
#         assert context


def test_reuse_context(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
    monkeypatch.chdir(fake_environment_directory)
    create_minimal_environment_yaml(fake_environment_directory / "environment.yml")
    with MlonMcuContext() as context:
        assert context
    with MlonMcuContext() as context2:
        assert context2


def test_reuse_context_locked(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
    monkeypatch.chdir(fake_environment_directory)
    create_minimal_environment_yaml(fake_environment_directory / "environment.yml")
    with MlonMcuContext() as context:
        assert context
    with MlonMcuContext() as context2:
        assert context2


def test_nest_context_read_after_read(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
    monkeypatch.chdir(fake_environment_directory)
    create_minimal_environment_yaml(fake_environment_directory / "environment.yml")
    with MlonMcuContext(deps_lock="read") as context:
        assert context
        with MlonMcuContext(deps_lock="read") as context2:
            assert context2


def test_nest_context_read_after_write(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
    monkeypatch.chdir(fake_environment_directory)
    create_minimal_environment_yaml(fake_environment_directory / "environment.yml")

    with MlonMcuContext(deps_lock="write") as context:
        assert context
        with pytest.raises(RuntimeError, match=r".*could\ not\ be\ acquired\..*"):
            with MlonMcuContext(deps_lock="read") as context2:
                assert context2


def test_nest_context_write_after_read(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
    monkeypatch.chdir(fake_environment_directory)
    create_minimal_environment_yaml(fake_environment_directory / "environment.yml")

    with MlonMcuContext(deps_lock="write") as context:
        assert context
        with pytest.raises(RuntimeError, match=r".*could\ not\ be\ acquired\..*"):
            with MlonMcuContext(deps_lock="read") as context2:
                assert context2


def test_nest_context_write_after_write(monkeypatch, fake_environment_directory: Path, fake_config_home: Path):
    monkeypatch.chdir(fake_environment_directory)
    create_minimal_environment_yaml(fake_environment_directory / "environment.yml")

    with MlonMcuContext(deps_lock="write") as context:
        assert context
        with pytest.raises(RuntimeError, match=r".*could\ not\ be\ acquired\..*"):
            with MlonMcuContext(deps_lock="read") as context2:
                assert context2
