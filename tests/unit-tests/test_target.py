import pytest
import mock

from mlonmcu.target.common import execute, cli
from mlonmcu.target.target import Target
from mlonmcu.target.etiss_pulpino import ETISSPulpinoTarget
from mlonmcu.target.host_x86 import HostX86Target

class CustomTarget(Target):

    def __init__(self, features=[], config={}, context=None):
        super().__init__("custom", features=features, config=config, context=context)
        self.inspectProgram = "which"
        self.inspectprogramArgs = []

    def exec(self, program, *args, **kwargs):
        return execute(program, *args, **kwargs)

@pytest.mark.parametrize("ignore_output", [False, True])
def test_target_common_execute(ignore_output, capsys):
    # TODO: mock command line!
    result = execute("date", ignore_output=ignore_output, live=False)
    out, err = capsys.readouterr()
    if ignore_output:
        assert len(out) == 0
    else:
        assert len(out) > 0

def test_target_common_cli_execute(capsys):
    cli(target=CustomTarget, args=["exec", "date", "-f", "featureA"])
    out, err = capsys.readouterr()

def _fake(*args, **kwargs):
    print("FAKE")

# @mock.patch('subprocess.run', side_effect=_fake)  # FIXME
def test_target_common_cli_inspect(capsys):
    cli(target=CustomTarget, args=["inspect", "date", "-c", "foo=bar"])
    out, err = capsys.readouterr()


@mock.patch('mlonmcu.target.common.execute', side_effect=_fake)  # FIXME
def test_target_custom(mocked_execute):
    t = CustomTarget()
    assert str(t) == "Target(custom)"

    # with pytest.raises(NotImplementedError):
    t.exec("date")

    t.inspect("date")
    # mocked_execute.assert_called_once_with("inspect", "program")

@mock.patch('mlonmcu.target.common.execute', side_effect=_fake)  # FIXME
def test_target_base(mocked_execute):
    t = Target("base")
    assert str(t) == "Target(base)"

    with pytest.raises(NotImplementedError):
        t.exec("date")

    # t.inspect("date")
    # mocked_execute.assert_called_once_with("inspect", "program")

def has_etiss_pulpino():
    return False

def has_riscv():
    return False

# TODO: needs etiss and riscv
@pytest.mark.skipif(not has_etiss_pulpino(), reason="requires etiss_pulpino")
@pytest.mark.skipif(not has_riscv(), reason="requires riscv")
@pytest.mark.parametrize("features", [[], ["etissdbg"], ["attach"], ["noattach"], ["trace"], ["v"]])
def test_target_etiss_pulpino(features):
    t = ETISSPulpinoTarget(config={"riscv.dir": "foo", "etiss.dir": "bar"}, features=features)
    # t.exec("date")

    # t.inspect("date")

# TODO: needs gdbserver
def has_gdbserver():
    return False

def has_gdb():
    return False

@pytest.mark.skipif(not has_gdbserver(), reason="requires gdbserver")
@pytest.mark.skipif(not has_gdb(), reason="requires gdb")
@pytest.mark.parametrize("features", [[], ["attach"], ["noattach"]])
def test_target_host_x86(features):
    t = HostX86Target(features=features)
    t.exec("date")

    # t.inspect("date")
