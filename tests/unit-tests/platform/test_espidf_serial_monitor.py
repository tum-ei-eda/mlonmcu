from types import SimpleNamespace

import pytest

from mlonmcu.platform.espidf.espidf import EspIdfPlatform


@pytest.mark.parametrize("complete", [True, False])
def test_serial_monitor_completion_and_timeout(tmp_path, monkeypatch, complete):
    serial = pytest.importorskip("serial")
    lines = iter([b"MLonMCU: START\n", b"result\n", b"MLonMCU: STOP\n"] if complete else [])
    closed = []

    class FakeSerial:
        def __init__(self, port, baud, timeout):
            assert timeout == 1

        def open(self):
            pass

        def close(self):
            closed.append(True)

        def readline(self):
            return next(lines, b"")

    monkeypatch.setattr(serial, "Serial", FakeSerial)
    monkeypatch.setattr("mlonmcu.platform.espidf.espidf.time.sleep", lambda _: None)
    clock = iter([0.0] + ([0.0] * 3 if complete else [2.0]))
    monkeypatch.setattr("mlonmcu.platform.espidf.espidf.time.monotonic", lambda: next(clock))
    platform = EspIdfPlatform(
        config={
            "espidf.src_dir": tmp_path,
            "espidf.install_dir": tmp_path,
            "espidf.use_idf_monitor": False,
            "espidf.port": "/dev/fake",
        }
    )
    target = SimpleNamespace(name="esp32p4", port=None, baud=None)
    if complete:
        assert "result\nMLonMCU: STOP" in platform.monitor(target, timeout=1)
    else:
        with pytest.raises(TimeoutError, match="UART monitor timed out"):
            platform.monitor(target, timeout=1)
    assert len(closed) == 2
