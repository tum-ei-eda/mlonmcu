import os
import pytest
import mock
import urllib.request

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

import mlonmcu  # TODO: fix this bad?

@pytest.fixture()
def fake_context():
    class FakeTaskCache():
        def __init__(self):
            self._vars = {}
    class FakeEnvironment():
        def __init__(self):
            self.paths = {}  # TODO: get rid of PathConfig if possible?
    class FakeContext():
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
