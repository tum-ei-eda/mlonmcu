from ..platform import BuildPlatform
from .microtvm_backend import create_microtvm_platform_backend, get_microtvm_platform_backends


class MicroTvmBuildPlatform(BuildPlatform):
    """MicroTVM build platform class."""

    FEATURES = BuildPlatform.FEATURES + []

    DEFAULTS = {
        **BuildPlatform.DEFAULTS,
    }

    REQUIRED = BuildPlatform.REQUIRED + []

    def create_backend(self, name):
        supported = self.get_supported_backends()
        assert name in supported, f"{name} is not a valid MicroTVM platform backend"
        base = supported[name]
        return create_microtvm_platform_backend(name, self, base=base)

    def get_supported_backends(self):
        backend_names = get_microtvm_platform_backends()
        return backend_names
