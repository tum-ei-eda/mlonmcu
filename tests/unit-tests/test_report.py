import pandas as pd
import pytest

from mlonmcu.report import Report


def test_report_set_combines_sections():
    report = Report()
    report.set([{"Model": "a"}], [{"Cycles": 10}], [{"Valid": True}])
    assert report.df.to_dict("records") == [{"Model": "a", "Cycles": 10, "Valid": True}]


def test_report_set_accepts_an_empty_main_section():
    report = Report()
    report.set([{"Model": "a"}], [], [{"Valid": True}])
    assert list(report.df.columns) == ["Model", "Valid"]


def test_report_add_accepts_one_or_many_reports():
    first, second, combined = Report(), Report(), Report()
    first.set_pre([{"id": 1}])
    second.set_pre([{"id": 2}])
    combined.add(first)
    combined.add([second])
    assert combined.pre_df.to_dict("records") == [{"id": 1}, {"id": 2}]


def test_report_exports_csv_and_creates_parent(tmp_path):
    report = Report()
    report.set_pre([{"id": 1}])
    path = tmp_path / "nested" / "report.csv"
    report.export(path)
    assert pd.read_csv(path).to_dict("records") == [{"id": 1}]


def test_report_rejects_unsupported_export_format(tmp_path):
    with pytest.raises(AssertionError, match="Unsupported report format"):
        Report().export(tmp_path / "report.json")
