# import onnxruntime
# import os
from onnxruntime.quantization import CalibrationDataReader
import numpy as np


class DummyDataReader(CalibrationDataReader):
    def __init__(self):
        self.datasize = 1
        self.first = True

    def get_next(self):
        if self.first:
            self.first = False
            # return {"input": np.random.uniform(-5.0, 5.0, [1, 3, 32, 32]).astype("float32")}
            return {"serving_default_input:0": np.random.uniform(-5.0, 5.0, [1, 1960]).astype("float32")}
        # if self.enum_data is None:
        #     pass
        return None

    def rewind(self):
        self.first = True
