import hashlib

import pytest
import yaml

from mlonmcu.models.frontend import ONNXFrontend, TfLiteFrontend
from mlonmcu.models.model import Model
from mlonmcu.setup.utils import validate_checksum


@pytest.mark.parametrize("frontend_cls", [TfLiteFrontend, ONNXFrontend])
@pytest.mark.parametrize("algorithm", ["sha1", "sha256", "md5"])
def test_frontend_checksum(tmp_path, frontend_cls, algorithm):
    path = tmp_path / "model.bin"
    path.write_bytes(b"expected model")
    definition = tmp_path / "definition.yml"
    definition.write_text(
        yaml.safe_dump(
            {
                "network": {
                    "hash": {
                        "algorithm": algorithm,
                        "value": hashlib.new(algorithm, path.read_bytes()).hexdigest(),
                    }
                }
            }
        )
    )
    model = Model("model", path, config={"model.metadata_path": str(definition)})
    frontend = frontend_cls()
    frontend.process_metadata(model, cfg={})
    path.write_bytes(b"modified model")
    with pytest.raises(RuntimeError, match="Checksum missmatch"):
        frontend.process_metadata(model, cfg={})
    frontend = frontend_cls(config={f"{frontend.name}.check_integrity": "False"})
    frontend.process_metadata(model, cfg={})


@pytest.mark.parametrize("network", [{}, {"filename": "model.bin"}, {"hash": None}])
def test_frontend_without_checksum(tmp_path, network):
    path = tmp_path / "model.bin"
    path.write_bytes(b"model without checksum")
    model = Model("model", path)
    model.metadata = {"network": network}
    TfLiteFrontend().process_metadata(model, cfg={})


@pytest.mark.parametrize("prefixed", [False, True])
def test_validate_sha1_checksum(tmp_path, prefixed):
    path = tmp_path / "model.bin"
    path.write_bytes(b"abc")
    checksum = "a9993e364706816aba3e25717850c26c9cd0d89d"
    if prefixed:
        assert validate_checksum(path, f"sha1:{checksum}")
    else:
        assert validate_checksum(path, checksum, mode="sha1")
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Checksum missmatch"):
        validate_checksum(path, checksum, mode="sha1")
