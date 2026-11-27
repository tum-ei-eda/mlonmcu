import pytest

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.session.postprocess import get_postprocesses
from mlonmcu.session.postprocess.rvv_bench import main, parse_rvv_bench, render_rvv_bench

SCALAR = """Boot noise
MLonMCU: START
<tr><td>add t0,t1,t2</td><td>1.03</td></tr>
<tr><td>slt &lt;t0&gt;</td><td>2.50</td></tr>
# Total Cycles: 1234
MLONMCU EXIT: 0
MLonMCU: STOP
"""


def test_scalar_report_filters_noise_and_escapes_metadata():
    html = render_rvv_bench(SCALAR, title="<script>bad</script>", comment="<unsafe>")
    assert "Cycles / instruction" in html
    assert html.count("<tbody>") == 1
    assert "<td>add t0,t1,t2</td><td>1.03</td>" in html
    assert "slt &lt;t0&gt;" in html
    assert "Boot noise" not in html
    assert "Total Cycles" not in html
    assert "<unsafe>" not in html
    assert "<script>" not in html
    assert 'src="../' not in html
    assert 'href="../' not in html


def test_vector_groups_ignore_iss_noise():
    text = """ISS start
vl=1 tu mu
<tr><td>vadd.vv</td><td>1</td><td></td></tr>
some diagnostic
<tr><td>vsub.vv</td><td>2</td><td>3</td></tr>
vl=VLMAX ta ma
<tr><td>vmul.vv</td><td>4</td><td>5</td></tr>
"""
    groups = parse_rvv_bench(text)
    assert list(groups) == ["vl=1 tu mu", "vl=VLMAX ta ma"]
    assert len(groups["vl=1 tu mu"]) == 2
    html = render_rvv_bench(text, vlen="128")
    assert html.count("<table>") == 2
    assert "VLEN=128" in html


def test_pie_methodology_metadata():
    text = "# Benchmark Info: PIE v1; median of 5 samples; <IRAM>\r\n" + SCALAR
    html = render_rvv_bench(text)
    assert "PIE v1; median of 5 samples; &lt;IRAM&gt;" in html


def test_pie_groups_and_sequence_units():
    text = """# Benchmark Unit: Cycles / test iteration
# Benchmark Group: Independent-input instructions
<tr><td>esp.vmul.s16 q2,q0,q1</td><td>1.03</td></tr>
# Benchmark Group: Dependent instruction chains
<tr><td>esp.vmul.s16 q2,q2,q1</td><td>2.03</td></tr>
# Benchmark Group: Producer/consumer sequences
<tr><td>load -&gt; add [2 instructions/iteration]</td><td>3.03</td></tr>
"""
    groups = parse_rvv_bench(text)
    assert list(groups) == [
        "Independent-input instructions",
        "Dependent instruction chains",
        "Producer/consumer sequences",
    ]
    html = render_rvv_bench(text)
    assert html.count("<table>") == 3
    assert html.count("Cycles / test iteration") == 3
    assert "Cycles / instruction" not in html
    assert "load -&gt; add [2 instructions/iteration]" in html


@pytest.mark.parametrize("text", ["noise", "<tr><td>add</td>", "<tr><td>add</td><td>nan</td></tr>"])
def test_missing_or_invalid_rows(text):
    with pytest.raises(ValueError, match="No complete"):
        render_rvv_bench(text)


@pytest.mark.parametrize("archived", [False, True])
def test_registered_postprocess(tmp_path, archived):
    if archived:
        log = tmp_path / "esp32p4_out.log"
        log.write_text(SCALAR)
        artifact = Artifact(log.name, path=log, fmt=ArtifactFormat.PATH)
    else:
        artifact = Artifact("etiss_out.log", content=SCALAR, fmt=ArtifactFormat.TEXT)
    processor = get_postprocesses()["rvv_bench"](config={"rvv_bench.comment": "P4 revision 1.3"})
    artifacts = processor.post_run(None, [artifact])
    assert artifacts[0].name == "rvv_bench.html"
    assert "P4 revision 1.3" in artifacts[0].content
    assert "1.03" in artifacts[0].content


def test_no_runtime_log():
    with pytest.raises(ValueError, match="runtime log"):
        get_postprocesses()["rvv_bench"]().post_run(None, [])


def test_cli(tmp_path, capsys):
    log = tmp_path / "out.log"
    log.write_text(SCALAR)
    main([str(log), "--scalar"])
    assert "Cycles / instruction" in capsys.readouterr().out
    output = tmp_path / "out.html"
    main([str(log), "--output", str(output)])
    assert "add t0,t1,t2" in output.read_text()
