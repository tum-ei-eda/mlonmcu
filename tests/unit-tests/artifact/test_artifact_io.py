from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlonmcu.artifact import Artifact, ArtifactFormat, lookup_artifacts


@pytest.mark.parametrize(
    "fmt,kwargs",
    [
        (ArtifactFormat.TEXT, {"content": None}),
        (ArtifactFormat.RAW, {"raw": None}),
        (ArtifactFormat.ARCHIVE, {"raw": None}),
        (ArtifactFormat.PATH, {"path": None}),
        (ArtifactFormat.JSON, {"data": {}}),
    ],
)
def test_artifact_validation_rejects_invalid_values(fmt, kwargs):
    error = NotImplementedError if fmt == ArtifactFormat.JSON else AssertionError
    with pytest.raises(error):
        Artifact("invalid", fmt=fmt, **kwargs)


def test_artifact_serialization_copy_and_representation(tmp_path):
    path = tmp_path / "artifact.txt"
    artifact = Artifact("artifact.txt", content="hello", fmt=ArtifactFormat.TEXT, flags={"log"}, optional=True)
    summary = artifact.serialize(full=True)
    assert summary == {
        "name": "artifact.txt",
        "path": None,
        "fmt": ArtifactFormat.TEXT.value,
        "flags": ["log"],
        "archive": False,
        "optional": True,
        "content": "hello",
        "data": None,
        "raw": None,
    }
    assert "artifact.txt" in repr(artifact)
    duplicate = artifact.copy()
    duplicate.path = path
    assert artifact.path is None


def test_lookup_artifacts_accepts_path_name():
    artifact = Artifact("result.txt", content="ok", fmt=ArtifactFormat.TEXT)
    assert lookup_artifacts([artifact], name=Path("result.txt")) == [artifact]


def test_export_text_and_raw_artifacts(tmp_path):
    text = Artifact("message.txt", content="hello", fmt=ArtifactFormat.TEXT)
    text.export(tmp_path)
    assert (tmp_path / "message.txt").read_text(encoding="utf-8") == "hello"
    assert text.exported is True

    raw = Artifact("message.bin", raw=b"\x00\x01", fmt=ArtifactFormat.RAW)
    destination = tmp_path / "renamed.bin"
    raw.export(destination)
    assert destination.read_bytes() == b"\x00\x01"


def test_export_skips_existing_artifact_unless_requested(tmp_path):
    path = tmp_path / "message.txt"
    artifact = Artifact("message.txt", content="first", fmt=ArtifactFormat.TEXT)
    artifact.export(tmp_path)
    artifact.content = "second"
    artifact.export(tmp_path)
    assert path.read_text(encoding="utf-8") == "first"
    artifact.export(tmp_path, skip_exported=False)
    assert path.read_text(encoding="utf-8") == "second"


def test_export_archive_can_extract(tmp_path, monkeypatch):
    extract = MagicMock()
    monkeypatch.setattr("mlonmcu.artifact.utils.extract", extract)
    artifact = Artifact("bundle.tar", raw=b"archive", fmt=ArtifactFormat.ARCHIVE)
    artifact.export(tmp_path, extract=True)
    extract.assert_called_once_with(tmp_path / "bundle.tar", tmp_path)


def test_export_path_artifact_uses_copy(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"data")
    copy = MagicMock()
    monkeypatch.setattr("mlonmcu.artifact.utils.copy", copy)
    Artifact("copied.bin", path=source, fmt=ArtifactFormat.PATH).export(tmp_path)
    copy.assert_called_once_with(source, tmp_path / "copied.bin")


def test_uncache_removes_in_memory_values(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"data")
    artifact = Artifact("artifact.bin", path=path, raw=b"data", fmt=ArtifactFormat.RAW)
    artifact.data = {"value": 1}
    artifact.content = "text"
    artifact.uncache()
    assert artifact.content is artifact.data is artifact.raw is None

    with pytest.raises(AssertionError, match="written to disk"):
        Artifact("raw.bin", raw=b"data", fmt=ArtifactFormat.RAW).uncache()


@pytest.mark.parametrize(
    "artifact,expected",
    [
        (Artifact("text", content="hello", fmt=ArtifactFormat.TEXT), "Content:"),
        (Artifact("raw", raw=b"12", fmt=ArtifactFormat.RAW), "Data Size: 2B"),
        (Artifact("archive", raw=b"123", fmt=ArtifactFormat.ARCHIVE), "Archive Size: 3B"),
    ],
)
def test_print_summary(artifact, expected, capsys):
    artifact.print_summary()
    assert expected in capsys.readouterr().out


def test_path_conversion_reads_text_and_binary(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("hello", encoding="utf-8")
    artifact = Artifact("input.txt", path=path, fmt=ArtifactFormat.PATH)
    assert artifact.convert(ArtifactFormat.TEXT).content == "hello"
    assert artifact.convert(ArtifactFormat.RAW).raw == b"hello"
