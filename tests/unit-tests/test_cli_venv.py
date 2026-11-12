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
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlonmcu.cli.main import maybe_reexec_in_workspace_venv


@pytest.mark.parametrize("already_active", [False, True])
def test_workspace_venv_detection_uses_prefix(tmp_path, monkeypatch, already_active):
    workspace = tmp_path / "workspace-venv"
    (workspace / "bin").mkdir(parents=True)
    (workspace / "bin" / "python").symlink_to(Path(sys.executable).resolve())
    environment = SimpleNamespace(python=SimpleNamespace(venv=workspace))
    monkeypatch.delenv("MLONMCU_VENV_REEXEC", raising=False)
    monkeypatch.setattr("mlonmcu.cli.main.resolve_environment_file", lambda: tmp_path / "environment.yml")
    monkeypatch.setattr("mlonmcu.cli.main.UserEnvironment.from_file", lambda path: environment)
    monkeypatch.setattr(sys, "prefix", str(workspace if already_active else tmp_path / "other-venv"))
    execve = Mock()
    monkeypatch.setattr("mlonmcu.cli.main.os.execve", execve)

    maybe_reexec_in_workspace_venv(["flow", "--help"])

    if already_active:
        execve.assert_not_called()
    else:
        execve.assert_called_once()
        executable, argv, env = execve.call_args.args
        assert executable == str(workspace / "bin" / "python")
        assert argv == [executable, "-m", "mlonmcu.cli.main", "flow", "--help"]
        assert env["MLONMCU_VENV_REEXEC"] == "1"
