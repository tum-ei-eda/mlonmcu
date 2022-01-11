from .codegen_wrapper import CodegenWrapper


class AOTWrapper(CodegenWrapper):
    def __init__(self, modelInfo, TODO):
        super().__init__(modelInfo)
        pass

    def get_wrapper(self):
        raise NotImplementedError
        # TODO
