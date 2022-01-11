class BackendModelOptions:
    def __init__(self, backend, supported=True, options={}):
        self.backend = backend
        self.supported = supported
        self.options = options


class TFLMIModelOptions(BackendModelOptions):
    def __init__(
        self,
        backend,
        supported=True,
        arena_size=None,
        builtin_ops=None,
        custom_ops=None,
    ):
        super().__init__(backend, supported=supported)
        self.arena_size = arena_size
        self.builtin_ops = builtin_ops
        self.custom_ops = custom_ops


class TVMRTModelOptions(BackendModelOptions):
    def __init__(self, backend, supported=True, arena_size=None):
        super().__init__(backend, supported=supported)
        self.arena_size = arena_size


def parse_model_options_for_backend(backend, options):
    backend_types = {
        "tflmi": TFLMIModelOptions,
        "tvmrt": TVMRTModelOptions,
    }
    if backend in backend_types:
        backend_type = backend_types[backend]
    else:
        backend_type = BackendModelOptions

    backend_options = backend_type(backend)

    for key, value in options.items():
        setattr(backend_options, key, value)

    return backend_options
