from .backend import TFLiteBackend

class TFLMCBackend(TFLiteBackend):

    shortname = "tflmc"
    
    def generate_code(self):
        pass