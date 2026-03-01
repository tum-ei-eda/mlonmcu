from pathlib import Path

from mlonmcu.models.frontend import TorchFrontend
from mlonmcu.models.model import ModelFormats


def test_torch_frontend_lookup_local_python_file(tmp_path):
    model_file = tmp_path / "sample_model.py"
    model_file.write_text("print('hello')\n")

    frontend = TorchFrontend()
    hints = frontend.lookup_models([str(model_file)])

    assert len(hints) == 1
    assert hints[0].name == "sample_model"
    assert hints[0].formats == [ModelFormats.PYTHON]
    assert hints[0].paths == [model_file]


def test_torch_frontend_lookup_local_python_file_with_model_class(tmp_path):
    model_file = tmp_path / "sample_model.py"
    model_file.write_text("import torch\nclass QuantLinearTest(torch.nn.Module):\n    pass\n")

    frontend = TorchFrontend()
    hints = frontend.lookup_models([f"{model_file}:QuantLinearTest"])

    assert len(hints) == 1
    assert hints[0].name == "QuantLinearTest"
    assert hints[0].alt == "sample_model"
    assert hints[0].formats == [ModelFormats.PYTHON]
    assert hints[0].paths == [model_file]


def test_torch_frontend_lookup_registered_model_download(tmp_path, monkeypatch):
    frontend = TorchFrontend(config={"torch.download_dir": str(tmp_path), "torch.use_default_registry": False})

    monkeypatch.setattr(frontend, "_load_registry", lambda: {"mv2": "https://example.com/mv2.pte"})

    def _fake_download(url, dst):
        assert url == "https://example.com/mv2.pte"
        Path(dst).write_bytes(b"pte")

    monkeypatch.setattr(frontend, "_download_file", _fake_download)

    hints = frontend.lookup_models(["mv2"])

    assert len(hints) == 1
    assert hints[0].name == "mv2"
    assert hints[0].formats == [ModelFormats.PTE]
    assert hints[0].paths == [tmp_path / "mv2.pte"]
    assert (tmp_path / "mv2.pte").is_file()


def test_torch_frontend_lookup_registered_model_with_model_class(tmp_path, monkeypatch):
    frontend = TorchFrontend(config={"torch.download_dir": str(tmp_path), "torch.use_default_registry": False})

    monkeypatch.setattr(
        frontend,
        "_load_registry",
        lambda: {
            "QuantLinearTest": {
                "url": "https://example.com/aot_arm_compiler.py",
                "class": "QuantLinearTest",
            }
        },
    )

    def _fake_download(url, dst):
        assert url == "https://example.com/aot_arm_compiler.py"
        Path(dst).write_text("import torch\nclass QuantLinearTest(torch.nn.Module):\n    pass\n")

    monkeypatch.setattr(frontend, "_download_file", _fake_download)

    hints = frontend.lookup_models(["QuantLinearTest"])

    assert len(hints) == 1
    assert hints[0].name == "QuantLinearTest"
    assert hints[0].alt == "QuantLinearTest"
    assert hints[0].formats == [ModelFormats.PYTHON]
    assert hints[0].paths == [tmp_path / "aot_arm_compiler.py"]
