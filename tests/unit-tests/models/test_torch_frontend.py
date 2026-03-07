from types import SimpleNamespace

import pytest

from mlonmcu.models.frontend import TorchFrontend
from mlonmcu.models.model import ModelFormats


def test_torch_frontend_lookup_default_package_model_class():
    frontend = TorchFrontend()

    hints = frontend.lookup_models(["qlinear"])

    assert len(hints) == 1
    assert hints[0].name == "QuantLinearTest"
    assert hints[0].alt == "qlinear"
    assert hints[0].formats == [ModelFormats.PYTHON]
    assert hints[0].paths[0].name == "quant_linear_test.py"


def test_torch_frontend_lookup_builtin_model_by_class_name():
    frontend = TorchFrontend()

    hints = frontend.lookup_models(["QuantAddTest"])

    assert len(hints) == 1
    assert hints[0].name == "QuantAddTest"
    assert hints[0].alt == "QuantAddTest"
    assert hints[0].formats == [ModelFormats.PYTHON]


def test_torch_frontend_lookup_model_class_in_environment_models_dir(tmp_path):
    model_file = tmp_path / "my_models.py"
    model_file.write_text("import torch\nclass MyLinearTest(torch.nn.Module):\n    pass\n")
    context = SimpleNamespace(environment=SimpleNamespace(paths={"models": [tmp_path]}))

    frontend = TorchFrontend()
    hints = frontend.lookup_models(["MyLinearTest"], context=context)

    assert len(hints) == 1
    assert hints[0].name == "MyLinearTest"
    assert hints[0].alt == "MyLinearTest"
    assert hints[0].formats == [ModelFormats.PYTHON]
    assert hints[0].paths == [model_file]


def test_torch_frontend_rejects_file_based_lookup(tmp_path):
    model_file = tmp_path / "sample_model.py"
    model_file.write_text("import torch\nclass SampleModel(torch.nn.Module):\n    pass\n")

    frontend = TorchFrontend()
    with pytest.raises(RuntimeError, match="Could not find Torch model"):
        frontend.lookup_models([str(model_file)])


def test_torch_frontend_rejects_class_qualified_file_lookup(tmp_path):
    model_file = tmp_path / "sample_model.py"
    model_file.write_text("import torch\nclass SampleModel(torch.nn.Module):\n    pass\n")

    frontend = TorchFrontend()
    with pytest.raises(RuntimeError, match="Could not find Torch model"):
        frontend.lookup_models([f"{model_file}:SampleModel"])
