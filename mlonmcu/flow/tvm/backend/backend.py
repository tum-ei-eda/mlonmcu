from mlonmcu.flow.backend import Backend

class TVMBackend(Backend):

    def load(self, model):
        self.model = model