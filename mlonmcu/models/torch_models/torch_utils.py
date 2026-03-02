# TODO: license & docstring
import sys
import pickle
import inspect
from pathlib import Path
from typing import Union, Tuple, Optional, Any

import torch


def _load_python_module_model(model_file: str, example_inputs: Any) -> Optional[Tuple[torch.nn.Module, Any]]:
    """Load a model and inputs from a Python source file.

    The file must define `ModelUnderTest` and `ModelInputs` attributes.

    Disclaimer: inspired by executorch/examples/arm/aot_arm_compiler.py

    """
    model_file = str(model_file)
    assert model_file.endswith(".py")

    import importlib.util

    spec = importlib.util.spec_from_file_location("tmp_model", model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model file {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["tmp_model"] = module
    model = module.ModelUnderTest
    inputs = example_inputs if example_inputs is not None else module.ModelInputs

    return model, inputs


def _load_serialized_model(model_name: str, example_inputs: Any) -> Optional[Tuple[torch.nn.Module, Any]]:  # nosec B614
    """Load a serialized Torch model saved via `torch.save`.

    Disclaimer: inspired by executorch/examples/arm/aot_arm_compiler.py

    """
    if not model_name.endswith((".pth", ".pt")):
        return None

    logging.info(f"Load model file {model_name}")

    model = torch.load(model_name, weights_only=False)  # nosec B614 trusted inputs
    if example_inputs is None:
        raise RuntimeError(f"Model '{model_name}' requires input data specify --model_input <FILE>.pt")

    return model, example_inputs


def load_torch_model(model_file: Union[str, Path]):

    model_file = Path(model_file)
    assert model_file.is_file()
    suffix = model_file.suffix

    if suffix == ".py":  # python source
        model, example_inputs = _load_python_module_model(model_file, None)
    elif suffix in [".pkl", ".pickle"]:  # pickled
        with open(model_file, "rb") as f:
            model = pickle.load(f)
    elif suffix in [".pth", "pt"]:  # torch serialized
        model, example_inputs = _load_serialized_model(model_file, None)
    else:
        raise RuntimeError(f"Unsupported suffix: {suffix}")

    if inspect.isclass(model):
        model = model()
    else:
        model = model
    assert isinstance(model, torch.nn.Module)

    print("model", model, dir(model))
    example_inputs = getattr(model, "example_input", None)
    if example_inputs is None:
        raise RuntimeError("Model must provide example_input")
    exported_program = torch.export.export(model, example_inputs, strict=True)
    return model, exported_program
