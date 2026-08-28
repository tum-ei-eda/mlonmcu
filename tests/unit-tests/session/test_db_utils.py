import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from mlonmcu.session import db_utils


def test_connect_to_postgres_db_uses_environment(monkeypatch):
    connection = MagicMock()
    connect = MagicMock(return_value=connection)
    monkeypatch.setitem(sys.modules, "psycopg2", SimpleNamespace(connect=connect))
    monkeypatch.setenv("POSTGRES_USER", "user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "database")
    monkeypatch.setenv("POSTGRES_HOST", "db.example")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    assert db_utils.connect_to_postgres_db() is connection
    connect.assert_called_once_with(
        host="db.example", port="5433", dbname="database", user="user", password="secret"
    )


def test_connect_to_postgres_db_requires_password(monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(AssertionError):
        db_utils.connect_to_postgres_db()


def test_connect_to_garage_uses_environment(monkeypatch):
    s3 = MagicMock()
    client = MagicMock(return_value=s3)
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=client))
    monkeypatch.setenv("S3_ENDPOINT", "https://s3.example")
    monkeypatch.setenv("S3_KEY", "key")
    monkeypatch.setenv("S3_SECRET", "secret")
    monkeypatch.setenv("REGION", "test-region")
    assert db_utils.connect_to_garage() is s3
    client.assert_called_once_with(
        "s3",
        endpoint_url="https://s3.example",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        region_name="test-region",
    )


@pytest.mark.parametrize("missing", ["S3_KEY", "S3_SECRET"])
def test_connect_to_garage_requires_credentials(monkeypatch, missing):
    monkeypatch.setenv("S3_KEY", "key")
    monkeypatch.setenv("S3_SECRET", "secret")
    monkeypatch.delenv(missing)
    with pytest.raises(AssertionError):
        db_utils.connect_to_garage()


def test_upload_report_stores_csv_and_metadata():
    s3, cursor, connection = MagicMock(), MagicMock(), MagicMock()
    report_df = MagicMock()
    report_df.to_csv.return_value = "name,value\ndemo,1\n"
    db_utils.upload_report(s3, cursor, connection, 12, report_df)
    body = b"name,value\ndemo,1\n"
    s3.put_object.assert_called_once_with(Bucket="mlonmcu", Key="experiments/12/report.csv", Body=body)
    values = cursor.execute.call_args.args[1]
    assert values[:3] == (12, "report", len(body))
    assert json.loads(values[3]) == {"s3_key": "experiments/12/report.csv", "bucket": "mlonmcu"}
    connection.commit.assert_called_once()


def test_upload_artifact_stores_session_and_run_files(tmp_path):
    path = tmp_path / "output.bin"
    path.write_bytes(b"1234")
    s3, cursor, connection = MagicMock(), MagicMock(), MagicMock()
    db_utils.upload_artifact(s3, cursor, connection, 7, ("output.bin", str(path)))
    db_utils.upload_artifact(s3, cursor, connection, 7, ("trace.log", str(path)), run_idx=2)
    assert s3.upload_file.call_args_list == [
        call(str(path), "mlonmcu", "experiments/7/output.bin"),
        call(str(path), "mlonmcu", "experiments/7/runs/2/trace.log"),
    ]
    assert cursor.execute.call_args_list[1].args[1][1:3] == ("runs/2/trace.log", 4)
    assert connection.commit.call_count == 2


def test_upload_artifact_rejects_missing_file(tmp_path):
    with pytest.raises(AssertionError, match="Not a file"):
        db_utils.upload_artifact(MagicMock(), MagicMock(), MagicMock(), 1, ("missing", tmp_path / "missing"))


def test_push_session_uploads_report_and_all_artifacts(monkeypatch):
    cursor, connection, s3 = MagicMock(), MagicMock(), MagicMock()
    connection.cursor.return_value = cursor
    cursor.fetchone.return_value = (42,)
    monkeypatch.setattr(db_utils, "connect_to_postgres_db", MagicMock(return_value=connection))
    monkeypatch.setattr(db_utils, "connect_to_garage", MagicMock(return_value=s3))
    upload_report, upload_artifact = MagicMock(), MagicMock()
    monkeypatch.setattr(db_utils, "upload_report", upload_report)
    monkeypatch.setattr(db_utils, "upload_artifact", upload_artifact)
    report = SimpleNamespace(df=MagicMock())
    db_utils.push_session_to_mlonmcu_db(
        report,
        session_artifacts=[("session.log", "/tmp/session.log")],
        run_artifacts={3: [("run.log", "/tmp/run.log")]},
        config_hash="abc",
        label="experiment",
        timestamp="2026-01-02",
        tags={"z", "a"},
    )
    assert cursor.execute.call_args.args[1] == ("experiment", "2026-01-02", "abc", '["a", "z"]')
    upload_report.assert_called_once_with(s3, cursor, connection, 42, report.df)
    assert upload_artifact.call_args_list == [
        call(s3, cursor, connection, 42, ("session.log", "/tmp/session.log")),
        call(s3, cursor, connection, 42, ("run.log", "/tmp/run.log"), run_idx=3),
    ]
    cursor.close.assert_called_once()
    connection.close.assert_called_once()


def test_push_session_accepts_no_optional_artifacts_or_tags(monkeypatch):
    cursor, connection = MagicMock(), MagicMock()
    connection.cursor.return_value = cursor
    cursor.fetchone.return_value = (1,)
    monkeypatch.setattr(db_utils, "connect_to_postgres_db", MagicMock(return_value=connection))
    monkeypatch.setattr(db_utils, "connect_to_garage", MagicMock())
    monkeypatch.setattr(db_utils, "upload_report", MagicMock())
    upload_artifact = MagicMock()
    monkeypatch.setattr(db_utils, "upload_artifact", upload_artifact)
    db_utils.push_session_to_mlonmcu_db(SimpleNamespace(df=MagicMock()))
    assert cursor.execute.call_args.args[1][3] == "[]"
    upload_artifact.assert_not_called()
