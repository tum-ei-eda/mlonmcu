import pandas as pd
import yaml

from mlonmcu.cli.postprocess import get_run_artifacts, load_report
from mlonmcu.session.postprocess.postprocesses import AnalyseVivadoReportsPostprocess


def test_load_report_restores_report_sections(tmp_path):
    path = tmp_path / "report.csv"
    pd.DataFrame(
        {
            "Session": [1],
            "Run": [0],
            "Cycles": [42],
            "Features": ["[]"],
            "Config": ["{}"],
            "Comment": ["-"],
        }
    ).to_csv(path, index=False)

    report = load_report(path)

    assert list(report.pre_df.columns) == ["Session", "Run"]
    assert list(report.main_df.columns) == ["Cycles"]
    assert list(report.post_df.columns) == ["Features", "Config", "Comment"]


def test_get_run_artifacts_restores_flags(tmp_path):
    artifact_path = tmp_path / "vivado.rpt"
    artifact_path.touch()
    with open(tmp_path / "artifacts.yml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"artifacts": [{"name": "vivado.rpt", "path": str(artifact_path), "flags": ["vivado", "report"]}]},
            handle,
        )

    artifacts = get_run_artifacts(tmp_path)

    assert len(artifacts) == 1
    assert artifacts[0].path == artifact_path
    assert artifacts[0].flags == ("vivado", "report")


def test_vivado_metric_categories():
    postprocess = AnalyseVivadoReportsPostprocess(config={"analyse_vivado_reports.limit": "timing,power"})

    metrics = postprocess.resolve_limit(
        ["timing_met", "wns_ns", "tns_ns", "whs_ns", "power_total_w", "power_dynamic_w", "power_static_w"]
    )

    assert metrics == ["timing_met", "wns_ns", "tns_ns", "whs_ns", "power_total_w", "power_dynamic_w", "power_static_w"]
