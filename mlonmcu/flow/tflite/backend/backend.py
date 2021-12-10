from mlonmcu.flow.backend import Backend

class TFLiteBackend(Backend):

    def load(self, model):
        self.model = model