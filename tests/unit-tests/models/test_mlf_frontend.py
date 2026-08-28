import io
import json
import tarfile

import pytest

from mlonmcu.artifact import ArtifactFormat
from mlonmcu.models import MLFFrontend
from mlonmcu.models.model import Model, ModelFormats
from mlonmcu.flow.tvm.backend.tvmaot import TVMAOTBackend
from mlonmcu.flow.tvm.backend.tvmrt import TVMRTBackend


def make_mlf(path, executor="aot"):
    metadata = {
        "version": 7,
        "modules": {
            "default": {
                "model_name": "default",
                "executors": [executor],
                "memory": {
                    "functions": {
                        "main": [
                            {
                                "inputs": {"input": {"dtype": "float32", "size": 8}},
                                "outputs": {"output": {"dtype": "float32", "size": 4}},
                                "workspace_size_bytes": 32,
                            }
                        ],
                        "operator_functions": [],
                    }
                },
            }
        },
    }
    members = {"metadata.json": json.dumps(metadata).encode()}
    if executor == "graph":
        members.update(
            {
                "executor-config/graph/default.graph": b"{}",
                "parameters/default.params": b"params",
            }
        )
    with tarfile.open(path, "w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_mlf_frontend(tmp_path):
    path = tmp_path / "model.tar"
    make_mlf(path)
    model = Model("model", [path], formats=[ModelFormats.MLF])
    artifacts = MLFFrontend().generate_artifacts(model)["default"]
    assert len(artifacts) == 1
    assert artifacts[0].fmt == ArtifactFormat.MLF
    assert "model" in artifacts[0].flags


def test_mlf_backend_reuses_archive(tmp_path):
    path = tmp_path / "model.tar"
    make_mlf(path, executor="graph")
    backend = TVMRTBackend(config={"tvmrt.generate_wrapper": False})
    backend.load_model(path)
    artifacts, _ = backend.generate()
    assert [artifact.name for artifact in artifacts["default"]] == [
        "default.json",
        "default.tar",
        "default.graph",
        "default.params",
    ]


def test_mlf_executor_mismatch(tmp_path):
    path = tmp_path / "model.tar"
    make_mlf(path, executor="graph")
    backend = TVMAOTBackend(config={"tvmaot.generate_wrapper": False})
    with pytest.raises(RuntimeError, match="executor mismatch"):
        backend.load_model(path)
