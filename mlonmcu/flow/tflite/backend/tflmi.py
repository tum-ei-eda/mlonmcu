from .backend import TFLiteBackend

class TFLMIBackend(TFLiteBackend):

    shortname = "tflmi"

    def generate_code(self):
        pass