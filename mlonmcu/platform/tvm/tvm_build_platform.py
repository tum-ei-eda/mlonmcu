from ..platform import BuildPlatform
from .tvm_backend import create_tvm_platform_backend, get_tvm_platform_backends


class TvmBuildPlatform(BuildPlatform):
    """TVM build platform class."""

    FEATURES = BuildPlatform.FEATURES + []

    DEFAULTS = {
        **BuildPlatform.DEFAULTS,
    }

    REQUIRED = BuildPlatform.REQUIRED + []

    def create_backend(self, name):
        supported = self.get_supported_backends()
        assert name in supported, f"{name} is not a valid TVM platform backend"
        base = supported[name]
        return create_tvm_platform_backend(name, self, base=base)

    def get_supported_backends(self):
        backend_names = get_tvm_platform_backends()
        return backend_names
