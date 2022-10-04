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
import mock

from mlonmcu.target.common import execute, cli
from mlonmcu.target.target import Target
from mlonmcu.target import EtissPulpinoTarget, HostX86Target


class CustomTarget(Target):
    FEATURES = ["featureA"]

    def __init__(
        self,
        features=[],
        config={},
    ):
        super().__init__("custom", features=features, config=config)
        self.inspectProgram = "ls"
        self.inspectprogramArgs = []

    def exec(self, program, *args, **kwargs):
        return execute(program, *args, **kwargs)


# @pytest.mark.parametrize("ignore_output", [False, True])
# def test_target_common_execute(ignore_output, capsys):
#     # TODO: mock command line!
#     result = execute("/bin/date", ignore_output=ignore_output, live=False)
#     out, err = capsys.readouterr()
#     if ignore_output:
#         assert len(out) == 0
#     else:
#         assert len(out) > 0
#
# def test_target_common_cli_execute(capsys):
#     cli(target=CustomTarget, args=["exec", "/bin/date", "-f", "featureA"])
#     out, err = capsys.readouterr()
#
def _fake(*args, **kwargs):
    print("FAKE")


# @mock.patch('subprocess.run', side_effect=_fake)  # FIXME
@pytest.mark.parametrize("example_elf_file", ["elf-Linux-x64-bash"], indirect=True)
def test_target_common_cli_inspect(example_elf_file, capsys):
    cli(target=CustomTarget, args=["inspect", example_elf_file, "-c", "foo=bar"])
    out, err = capsys.readouterr()


@mock.patch("mlonmcu.target.common.execute", side_effect=_fake)  # FIXME
@pytest.mark.parametrize("example_elf_file", ["elf-Linux-x64-bash"], indirect=True)
def test_target_custom(mocked_execute, example_elf_file, capsys):
    t = CustomTarget()
    assert str(t) == "Target(custom)"

    # with pytest.raises(NotImplementedError):
    # t.exec("/bin/date")

    t.inspect(example_elf_file)
    # mocked_execute.assert_called_once_with("inspect", "program")


@mock.patch("mlonmcu.target.common.execute", side_effect=_fake)  # FIXME
@pytest.mark.parametrize("example_elf_file", ["elf-Linux-x64-bash"], indirect=True)
def test_target_base(mocked_execute, example_elf_file, capsys):
    t = Target("base")
    assert str(t) == "Target(base)"

    with pytest.raises(NotImplementedError):
        t.exec("/bin/date")

    t.inspect(example_elf_file)
    # mocked_execute.assert_called_once_with("inspect", "program")


def has_etiss_pulpino():
    return False


def has_riscv():
    return False


# TODO: needs etiss and riscv
@pytest.mark.skipif(not has_etiss_pulpino(), reason="requires etiss_pulpino")
@pytest.mark.skipif(not has_riscv(), reason="requires riscv")
@pytest.mark.parametrize("features", [[], ["etissdbg"], ["attach"], ["noattach"], ["trace"], ["v"]])
@pytest.mark.parametrize("example_elf_file", ["elf-Linux-ARM64-bash"], indirect=True)
def test_target_etiss_pulpino(features, example_elf_file, capsys):
    t = EtissPulpinoTarget(config={"riscv.dir": "foo", "etiss.dir": "bar"}, features=features)
    # t.exec("/bin/date")

    t.inspect(example_elf_file)


# TODO: needs gdbserver
def has_gdbserver():
    return False


def has_gdb():
    return False


@pytest.mark.skipif(not has_gdbserver(), reason="requires gdbserver")
@pytest.mark.skipif(not has_gdb(), reason="requires gdb")
@pytest.mark.parametrize("features", [[], ["attach"], ["noattach"]])
@pytest.mark.parametrize("example_elf_file", ["elf-Linux-x64-bash"], indirect=True)
def test_target_host_x86(features, example_elf_file, capsys):
    t = HostX86Target(features=features)
    t.exec("/bin/date")

    t.inspect(example_elf_file)
