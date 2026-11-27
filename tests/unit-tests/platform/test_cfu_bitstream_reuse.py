"""FPGA reuse control flow without Vivado or a physical board."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlonmcu.platform.cfu_playground.cfu_playground import CFUPlaygroundPlatform
from mlonmcu.platform.cfu_playground import bitstream


def setup(tmp_path, reuse=True):
    deps = tmp_path / "deps"
    (deps / "soc").mkdir(parents=True)
    (deps / "soc/common_soc.py").write_text("# --load-only")
    source = tmp_path / "old/cfu_playground/out/gateware/digilent_arty.bit"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"bitstream")
    platform = CFUPlaygroundPlatform(
        config={
            "cfu_playground.src_dir": deps,
            "mlif.src_dir": tmp_path,
            "cfu_playground.reuse_bitstream": reuse,
            "cfu_playground.bitstream_file": source if reuse else None,
            "cfu_playground.wait_for_user": False,
        }
    )
    platform.init_directory(tmp_path / "new")
    platform.prepare_environment = Mock(return_value={})
    target = SimpleNamespace(
        fpga_sim=True,
        rtl_sim=False,
        use_renode=False,
        use_sw_dir=None,
        use_gateware_dir=None,
        fpga_target="digilent_arty",
        fpga_variant="a7-100",
        cpu_variant="full+cfu",
        sys_clk_freq=75000000,
        integrated_main_ram_size=0,
        baud=1843200,
        fpga_tty="/dev/ttyUSB0",
        name="cfu_full_fpga",
    )
    return platform, target, source


def test_reuse_program_upload_metrics_and_artifacts(tmp_path, monkeypatch):
    platform, target, source = setup(tmp_path)
    make = Mock(side_effect=AssertionError("must not run make bitstream/prog"))
    execute = Mock(return_value="output")
    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.utils.make", make)
    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.utils.execute", execute)
    out, artifacts, metrics = platform.run(None, target)
    assert metrics.get("FPGA Bitstream Time [s]") == 0
    assert metrics.get("FPGA Bitstream Reused") is True
    assert metrics.has("FPGA Prog Time [s]") and metrics.has("FPGA Monitor Time [s]")
    assert "--load-only" in execute.call_args_list[0].args
    assert str(platform.out_dir / "gateware" / f"{target.fpga_target}.bit") in execute.call_args_list[0].args
    assert str(execute.call_args_list[1].args[0]).endswith("interact_mlonmcu.expect")
    meta = json.loads(next(a.content for a in artifacts if a.name == bitstream.MANIFEST))
    assert meta["source"] == str(source.resolve())
    assert meta["bitstream_sha256"] == bitstream.sha256(source)
    assert meta["reports"] == {}
    assert source.read_bytes() == b"bitstream"


@pytest.mark.parametrize("empty", [False, True])
def test_invalid_file(tmp_path, empty):
    platform, target, source = setup(tmp_path)
    source.write_bytes(b"") if empty else source.unlink()
    with pytest.raises(ValueError, match="missing or empty"):
        bitstream.resolve(platform, target)


@pytest.mark.parametrize("change", ["disabled", "no-file", "rtl", "shared-sw"])
def test_invalid_options(tmp_path, change):
    platform, target, _ = setup(tmp_path)
    if change == "disabled":
        platform.config["reuse_bitstream"] = False
    elif change == "no-file":
        platform.config["bitstream_file"] = None
    elif change == "rtl":
        target.rtl_sim = True
    else:
        target.use_sw_dir = "/shared"
    with pytest.raises(ValueError):
        bitstream.resolve(platform, target)


@pytest.mark.parametrize("field", bitstream.FIELDS)
def test_manifest_mismatch(tmp_path, field):
    platform, target, source = setup(tmp_path)
    (source.parent / bitstream.MANIFEST).write_text(
        json.dumps({"bitstream_sha256": bitstream.sha256(source), "hardware": {field: "wrong"}})
    )
    with pytest.raises(ValueError, match=field):
        bitstream.resolve(platform, target)


def test_cfu_and_csr_mismatch(tmp_path):
    platform, target, source = setup(tmp_path)
    project = source.parent.parent.parent
    (project / "cfu.v").write_text("old")
    (platform.project_dir / "cfu.v").write_text("new")
    with pytest.raises(ValueError, match="cfu_sha256"):
        bitstream.resolve(platform, target)
    (platform.project_dir / "cfu.v").write_text("old")
    (project / "out/csr.json").write_text("{}")
    state = bitstream.resolve(platform, target)
    with pytest.raises(ValueError, match="csr.json"):
        bitstream.check_generated(platform, state)


def test_software_compile_is_preserved(tmp_path, monkeypatch):
    platform, target, _ = setup(tmp_path)
    platform.prepare = Mock(return_value="")
    make = Mock(return_value="software")
    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.utils.make", make)
    assert platform.compile(target, src=tmp_path) == "software"
    assert make.call_args.args[0] == "software"


def test_disabled_generation_and_reports(tmp_path, monkeypatch):
    platform, target, source = setup(tmp_path, reuse=False)
    gateware = platform.out_dir / "gateware"
    gateware.mkdir(parents=True)
    (gateware / "digilent_arty.bit").write_bytes(b"new")
    for kind in ("timing", "power", "utilization_place", "utilization_synth", "utilization_hierarchical_place"):
        (gateware / f"arty_{kind}.rpt").write_text("report")
    make = Mock(return_value="generated")
    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.utils.make", make)
    _, artifacts, metrics = platform.synth(None, target)
    assert make.call_args.args[0] == "bitstream"
    assert len(artifacts) == 6
    assert metrics.get("FPGA Bitstream Reused") is False
    platform.flash(None, target)
    assert make.call_args.args[0] == "prog-only"
    assert f"BITSTREAM={gateware / 'digilent_arty.bit'}" in make.call_args.args


@pytest.mark.parametrize("board", ["digilent_arty", "other_board"])
def test_fresh_generation_programs_run_local_bitstream(tmp_path, monkeypatch, board):
    platform, target, _ = setup(tmp_path, reuse=False)
    target.fpga_target = board
    path = platform.out_dir / "gateware" / f"{board}.bit"

    def make(*args, **kwargs):
        if args[0] == "bitstream":
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fresh bitstream")
            for kind in ("timing", "power", "utilization_place", "utilization_synth", "utilization_hierarchical_place"):
                (path.parent / f"{board}_{kind}.rpt").write_text("report")
        else:
            assert args[0] == "prog-only"
            assert f"BITSTREAM={path}" in args
            assert path.read_bytes() == b"fresh bitstream"
        return "output"

    mocked_make = Mock(side_effect=make)
    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.utils.make", mocked_make)
    platform.synth(None, target)
    platform.flash(None, target)
    assert [call.args[0] for call in mocked_make.call_args_list] == ["bitstream", "prog-only"]


@pytest.mark.parametrize("empty", [False, True])
def test_fresh_program_requires_bitstream(tmp_path, monkeypatch, empty):
    platform, target, _ = setup(tmp_path, reuse=False)
    path = platform.out_dir / "gateware" / f"{target.fpga_target}.bit"
    if empty:
        path.parent.mkdir(parents=True)
        path.touch()
    make = Mock()
    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.utils.make", make)
    with pytest.raises(ValueError, match="Run-local bitstream is missing or empty") as exc:
        platform.flash(None, target)
    assert str(path) in str(exc.value)
    make.assert_not_called()


def test_reports_and_independent_copies(tmp_path):
    platform, target, source = setup(tmp_path)
    report = source.parent / "arty_timing.rpt"
    report.write_text("original")
    _, artifacts, _ = platform.synth(None, target)
    assert any(a.name == "vivado_report_timing.rpt" for a in artifacts)
    other = tmp_path / "other"
    platform.project_dir = other
    other.mkdir()
    platform.synth(None, target)
    (other / "out/gateware/digilent_arty.bit").write_bytes(b"changed")
    assert (tmp_path / "new/out/gateware/digilent_arty.bit").read_bytes() == source.read_bytes()
    assert report.read_text() == "original"


def test_missing_reports_postprocess(tmp_path):
    import pandas as pd
    from mlonmcu.session.postprocess.postprocesses import AnalyseVivadoReportsPostprocess

    platform, target, _ = setup(tmp_path)
    _, artifacts, _ = platform.synth(None, target)
    report = SimpleNamespace(pre_df=pd.DataFrame({"Platform": ["cfu_playground"]}))
    assert AnalyseVivadoReportsPostprocess().post_run(report, artifacts) == []


def test_program_and_monitor_share_lock(tmp_path, monkeypatch):
    platform, target, _ = setup(tmp_path)
    held = []

    class Lock:
        def __init__(self, path):
            assert Path(path).name == "mlonmcu_serial.lock"

        def __enter__(self):
            held.append(True)

        def __exit__(self, *args):
            held.pop()

    def execute(*args, **kwargs):
        assert held == [True]
        return ""

    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.FileLock", Lock)
    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.utils.execute", execute)
    platform.run(None, target)
    assert held == []


def test_directory_source_and_matching_manifest(tmp_path):
    platform, target, source = setup(tmp_path)
    meta = bitstream.provenance(platform, target, source)
    (source.parent / bitstream.MANIFEST).write_text(json.dumps(meta))
    platform.config["bitstream_file"] = source.parents[3]
    state = bitstream.resolve(platform, target)
    assert state["compatibility"] == "partial-user-responsibility"
    source.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        bitstream.resolve(platform, target)


def test_report_device_mismatch(tmp_path):
    platform, target, source = setup(tmp_path)
    (source.parent / "arty_timing.rpt").write_text("| Device : xc7a35ticsg324-1L |")
    with pytest.raises(ValueError, match="FPGA device"):
        bitstream.resolve(platform, target)


def bios(timestamp, payload=b"code", valid_crc=True):
    import zlib

    data = payload + b"BIOS built on " + timestamp + b"\0"
    return data + (zlib.crc32(data).to_bytes(4, "little") if valid_crc else b"bad!")


def generated_pair(platform, source, name, old, new):
    original = source.parents[1] / name
    current = platform.out_dir / name
    original.parent.mkdir(parents=True, exist_ok=True)
    current.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(old)
    current.write_bytes(new)
    return original, current


@pytest.mark.parametrize("history", ["none", "v1", "v2", "v2-without-source"])
def test_normalized_reuse_program_upload(tmp_path, monkeypatch, history):
    platform, target, source = setup(tmp_path)
    pairs = {
        bitstream.BIOS: (bios(b"Nov 26 2026 14:54:42"), bios(b"Nov 26 2026 16:06:21")),
        "csr.json": (b'{"csr_bases":{"uart":100},"memories":{}}', b'{ "memories": {}, "csr_bases": {"uart": 100}}'),
        "software/include/generated/csr.h": (
            b"// Auto-generated by LiteX (8c889eacd) on 2026-11-26 14:54:42\n#define CSR 100\n",
            b"// Auto-generated by LiteX (8c889eacd) on 2026-11-26 16:06:21\n#define CSR 100\n",
        ),
        "software/include/generated/variables.mak": (
            b"CPU=vexriscv\nBUILDINC_DIRECTORY=/old/run/include\n",
            b"CPU=vexriscv\nBUILDINC_DIRECTORY=/new/run/include\n",
        ),
        "software/include/generated/regions.ld": (b"rom : ORIGIN = 0, LENGTH = 100",) * 2,
    }
    originals = [generated_pair(platform, source, name, *data)[0] for name, data in pairs.items()]
    if history != "none":
        meta = bitstream.provenance(platform, target, source)
        out = source.parents[1]
        meta["generated"] = bitstream.generated_hashes(out)
        meta["generated_compatibility"] = bitstream.generated_compatibility(out)
        if history == "v1":
            meta["schema_version"] = 1
            del meta["generated_compatibility"]
            del meta["generated"]["software/include/generated/variables.mak"]
        (source.parent / bitstream.MANIFEST).write_text(json.dumps(meta))
    state = bitstream.resolve(platform, target)
    if history == "v2-without-source":
        for original in originals:
            original.unlink()
    bitstream.check_generated(platform, state)
    make = Mock(side_effect=AssertionError("FPGA implementation must not run"))
    execute = Mock(return_value="")
    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.utils.make", make)
    monkeypatch.setattr("mlonmcu.platform.cfu_playground.cfu_playground.utils.execute", execute)
    platform.run(None, target)
    assert "--load-only" in execute.call_args_list[0].args
    assert str(execute.call_args_list[1].args[0]).endswith("interact_mlonmcu.expect")
    make.assert_not_called()
    for original, data in zip(originals, pairs.values()):
        if original.exists():
            assert original.read_bytes() == data[0]


@pytest.mark.parametrize(
    "name,old,new",
    [
        ("csr.json", b'{"csr_bases":{"uart":100}}', b'{"csr_bases":{"uart":104}}'),
        ("csr.json", b'{"memories":{"rom":{"size":100}}}', b'{"memories":{"rom":{"size":200}}}'),
        ("software/include/generated/regions.ld", b"ORIGIN = 0", b"ORIGIN = 4"),
        ("software/include/generated/csr.h", b"#define CSR 100", b"#define CSR 104"),
        ("software/include/generated/variables.mak", b"CPU=old", b"CPU=new"),
    ],
)
def test_semantic_generated_changes_rejected(tmp_path, name, old, new):
    platform, target, source = setup(tmp_path)
    generated_pair(platform, source, name, old, new)
    with pytest.raises(ValueError, match="generated LiteX metadata"):
        bitstream.check_generated(platform, bitstream.resolve(platform, target))


@pytest.mark.parametrize("change", ["code", "checksum"])
def test_unknown_bios_difference_warns_or_fails_strict(tmp_path, caplog, change):
    platform, target, source = setup(tmp_path)
    old = bios(b"Nov 26 2026 14:54:42")
    new = bios(
        b"Nov 26 2026 16:06:21", payload=b"changed" if change == "code" else b"code", valid_crc=change != "checksum"
    )
    generated_pair(platform, source, bitstream.BIOS, old, new)
    state = bitstream.resolve(platform, target)
    bitstream.check_generated(platform, state)
    assert "BIOS differs beyond recognized normalization" in caplog.text
    platform.config["strict_bios_compatibility"] = True
    with pytest.raises(ValueError, match="BIOS differs beyond"):
        bitstream.check_generated(platform, state)
