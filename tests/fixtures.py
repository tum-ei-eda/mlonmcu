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
import os
import pytest
import mock
import tempfile
import urllib.request
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

# import mlonmcu  # TODO: fix this bad?


@pytest.fixture()
def fake_config_home(tmp_path):
    config_home = tmp_path / "cfg"
    config_home.mkdir()
    mlonmcu_config_home = config_home / "mlonmcu"
    mlonmcu_config_home.mkdir()
    patcher = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_home)})
    patcher.start()
    yield mlonmcu_config_home
    patcher.stop()


@pytest.fixture()
def fake_environment_directory(tmp_path):
    cwd = tmp_path / "home"
    cwd.mkdir()
    yield cwd


@pytest.fixture()
def fake_working_directory(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(str(cwd))
    yield cwd


@pytest.fixture()
def fake_context():
    class FakeTaskCache:
        def __init__(self):
            self._vars = {}

    class FakeEnvironment:
        def __init__(self):
            self.paths = {}  # TODO: get rid of PathConfig if possible?
            self.flags = {}
            self.vars = {}

    class FakeContext:
        def __init__(self):
            self.environment = FakeEnvironment()
            self.cache = FakeTaskCache()

    context = FakeContext()
    yield context


@pytest.fixture()
def example_elf_file(request, tmp_path):
    name = request.param
    elf_path = tmp_path / name
    url = f"https://github.com/JonathanSalwan/binary-samples/raw/master/{name}"
    urllib.request.urlretrieve(url, elf_path)
    yield str(elf_path)


@pytest.fixture()
def user_context():
    from mlonmcu.context.context import MlonMcuContext

    try:
        with MlonMcuContext(deps_lock="read") as context:
            yield context
    except RuntimeError:
        pytest.skip("User Environment not found!")


@pytest.fixture(scope="session")
def models_dir():
    with tempfile.TemporaryDirectory() as tmp_path:
        url = "https://codeload.github.com/tum-ei-eda/mlonmcu-models/zip/refs/heads/main"
        resp = urllib.request.urlopen(url)
        with ZipFile(BytesIO(resp.read())) as zip_file:
            zip_file.extractall(tmp_path)
        models_path = Path(tmp_path) / "mlonmcu-models-main"
        yield models_path
