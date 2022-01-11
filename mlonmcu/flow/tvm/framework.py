from mlonmcu.flow.framework import Framework
from mlonmcu.flow.tvm import TVMBackend


class TVMFramework(Framework):

    shortname = "tvm"

    def __init__(self):
        super().__init__()
        self.backends = TVMBackend.registry
