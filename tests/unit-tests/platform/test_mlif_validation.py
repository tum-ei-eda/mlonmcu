"""Regression coverage for the binary validation transport and batch handling."""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from mlonmcu.platform.mlif.interfaces import ModelSupport
from mlonmcu.platform.mlif.mlif import MlifPlatform
from mlonmcu.platform.mlif.mlif_target import create_mlif_platform_target


def test_auto_interfaces_and_rom_batch():
    target = SimpleNamespace(supports_stdin=True, supports_stdout=True, supports_filesystem=True)
    support = ModelSupport("auto", "auto", {}, target=target)
    assert (support.in_interface, support.out_interface) == ("stdin_raw", "stdout_raw")
    support = ModelSupport("rom", "stdout_raw", {}, batch_size=2, inputs_data=[{}, {}, {}])
    assert support.batch_size == 3
    with pytest.raises(ValueError, match="positive"):
        ModelSupport("rom", "stdout_raw", {}, batch_size=0)


def test_validation_selects_v2():
    platform = MlifPlatform(config={"mlif.src_dir": "/tmp", "mlif.set_inputs": True})
    assert platform.template_version == "v2"
    platform.config["template_version"] = "v1"
    with pytest.raises(ValueError, match="require"):
        platform.get_definitions()


@pytest.mark.parametrize("truncate", [False, True])
def test_binary_batches_multi_output_and_embedded_markers(tmp_path, truncate):
    # Both framing markers can occur in raw output bytes.
    payload = b"-?--!-"
    info = {
        "input_names": ["input"],
        "output_names": ["a", "b"],
        "output_types": ["uint8", "int16"],
        "output_shapes": [[6], [1]],
    }
    info_file = tmp_path / "model_info.yml"
    info_file.write_text(yaml.safe_dump(info))
    inputs_file = tmp_path / "inputs.npy"
    np.save(inputs_file, [{"input": np.array([i], dtype="uint8")} for i in range(5)])
    platform = SimpleNamespace(
        set_inputs=True,
        get_outputs=True,
        model_info_file=info_file,
        inputs_artifact=inputs_file,
        set_inputs_interface="stdin_raw",
        get_outputs_interface="stdout_raw",
        batch_size=2,
    )

    class Transport:
        DEFAULTS = {}
        calls = []

        def __init__(self, **kwargs):
            pass

        def exec(self, program, *args, stdin_data=None, **kwargs):
            self.calls.append(stdin_data)
            raw = b"log\n"
            for value in stdin_data:
                raw += b"-?-" + payload + b"-!-\n"
                raw += b"-?-" + np.array([value], dtype="int16").tobytes() + b"-!-\n"
            return raw[:-5] if truncate else raw, []

    target = create_mlif_platform_target("test", platform, base=Transport)()
    if truncate:
        with pytest.raises(ValueError, match="truncated"):
            target.exec("unused", cwd=tmp_path)
    else:
        _, artifacts = target.exec("unused", cwd=tmp_path)
        assert target.calls == [b"\0\1", b"\2\3", b"\4"]
        outputs = np.load(tmp_path / "outputs.npy", allow_pickle=True)
        assert len(outputs) == 5
        assert outputs[0]["a"].tobytes() == payload
        assert outputs[4]["b"].item() == 4
        assert artifacts[-1].name == "outputs.npy"
