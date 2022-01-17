from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Callable
import logging
import copy

from mlonmcu.models.model import Model, ModelFormat

from mlonmcu.logging import get_logger

logger = get_logger()


extension_to_fmt = {
    "tflite": ModelFormat.TFLITE,
    "onnx": ModelFormat.ONNX,
    "ipynb": ModelFormat.IPYNB,
    "tflm": ModelFormat.PACKED,
}


class FrontendRegistry:

    registry = {}
    supports = {}
    frontends_for_fmt = {}

    @classmethod
    def get_supported_frontends(cls, fmt):
        if fmt not in cls.frontends_for_fmt:
            return []
        return cls.frontends_for_fmt[fmt]

    @classmethod
    def register(cls, name: str, supports=[]) -> Callable:
        def inner_wrapper(wrapped_class: Frontend) -> Callable:
            if name in cls.registry:
                logger.warning("Frontend %s already exists. Will replace it", name)
            cls.registry[name] = wrapped_class
            cls.supports[name] = supports
            for s in supports:
                if s in cls.frontends_for_fmt:
                    if name not in cls.frontends_for_fmt[s]:
                        cls.frontends_for_fmt[s].append(wrapped_class)
                else:
                    cls.frontends_for_fmt[s] = [wrapped_class]

            # logger.debug('Registered frontend %s', name)
            print("Registered frontend", name)
            return wrapped_class

        return inner_wrapper


class Frontend(ABC):
    def __init__(self, name="", format=ModelFormat.NONE, features=[], cfg={}):
        self.name = name
        self.format = format
        self.features = features
        self.cfg = cfg

    def convert(self, model, to=ModelFormat.TFLITE):
        print(model.format)
        print("convert to", to)
        new_model = self._convert(model)
        print("new_model1", new_model)
        if new_model:
            if new_model.format == to:
                return new_model
            else:
                model = new_model
        print("B")
        frontends = FrontendRegistry.get_supported_frontends(model.format)
        for frontend in frontends:
            print("C", frontend)
            # input()
            new_model = frontend(features=self.features, cfg=self.cfg).convert(
                model, to=to
            )
            print("new_model", new_model, model)
            if new_model:
                if new_model.format == to:
                    return new_model
        return None  # TODO: error

    def load(self, file):
        print("load")
        if not isinstance(file, Path):
            file = Path(file)
        ext = file.suffix[1:]
        fmt = extension_to_fmt[ext]  # func with error check?
        assert fmt == self.format
        # model = self._load(file)
        model = Model(name="?", path=file, format=fmt)
        return model

    @abstractmethod
    def _convert(self, model):
        pass

    @abstractmethod
    def _load(self, file):
        pass

    @abstractmethod
    def _export(self, file):
        pass


@FrontendRegistry.register("onnx", supports=[ModelFormat.IPYNB])
class ONNXFrontend(Frontend):
    def __init__(self, features=[], cfg={}):
        super().__init__(
            name="onnx", format=ModelFormat.ONNX, features=features, cfg=cfg
        )

    def _convert(self, model):
        print("_convert onnx")
        from_fmt = model.format
        if from_fmt == ModelFormat.IPYNB:
            new_model = copy.deepcopy(model)
            new_model.format = ModelFormat.ONNX
            return new_model
        return None

    def _load(self, file):
        model = super()._load()
        # TODO: postprocessing (metadata?)
        return model


@FrontendRegistry.register("tflite", supports=[ModelFormat.IPYNB])
class TfLiteFrontend(Frontend):
    def __init__(self, features=[], cfg={}):
        super().__init__(
            name="tflite", format=ModelFormat.TFLITE, features=features, cfg=cfg
        )

    def _convert(self, model):
        print("_convert tflite")
        from_fmt = model.format
        if from_fmt == ModelFormat.IPYNB:
            new_model = copy.deepcopy(model)
            new_model.format = ModelFormat.TFLITE
            return new_model
        return None

    def _load(self, file):
        model = super()._load()
        # TODO: postprocessing (metadata?)
        return model


@FrontendRegistry.register("packed", supports=[ModelFormat.TFLITE])
class PackedFrontend(Frontend):
    def __init__(self, features=[], cfg={}):
        super().__init__(
            name="packed", format=ModelFormat.PACKED, features=features, cfg=cfg
        )

    def _convert(self, model):
        print("_convert packed")
        from_fmt = model.format
        if from_fmt == ModelFormat.TFLITE:
            new_model = copy.deepcopy(model)
            new_model.format = ModelFormat.PACKED
            return new_model
        return None

    def _load(self, file):
        model = super()._load()
        # TODO: postprocessing (metadata?)
        return model


@FrontendRegistry.register("ipynb", supports=[])
class IPYNBFrontend(Frontend):
    def __init__(self, features=[], cfg={}):
        super().__init__(
            name="ipynb", format=ModelFormat.IPYNB, features=features, cfg=cfg
        )

    def _convert(self, model):
        print("_convert ipynb")
        return

    def _load(self, file):
        model = super()._load()
        # TODO: postprocessing (metadata?)
        return model


print("ABC")
print(FrontendRegistry.registry)
print(FrontendRegistry.supports)
print(FrontendRegistry.frontends_for_fmt)
frontend = FrontendRegistry.registry["ipynb"]()
print(frontend)
model = frontend.load("abc.ipynb")
print(model.format)
converted_model = frontend.convert(model, to=ModelFormat.PACKED)
print(converted_model.format)

print("XYZ")
