#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import re
import os

from mlonmcu.models.model import ModelFormats


class TensorInfo:
    def __init__(self, name, shape, dtype, fix_names=False):
        self.name = name
        if fix_names:
            self.name = self.name.replace("/", "_").replace(";", "_")
        assert isinstance(shape, (tuple, list))
        self.shape = shape
        size_lookup = {
            "float32": 4,
            "uint8": 1,
            "int8": 1,
            "int32": 4,
            "int64": 8,
        }
        assert dtype in size_lookup, f"Unsupported type: {dtype}"
        self.dtype = dtype
        self.type_size = size_lookup[self.dtype]

    @property
    def size(self):
        ret = self.type_size
        for dim in self.shape:
            if isinstance(dim, complex):
                real = dim.real
                imag = dim.imag
                assert real == int(real)
                assert imag == int(imag)
                ret *= int(real) + int(imag)
            else:
                ret *= dim
        return ret


class TfLiteTensorInfo(TensorInfo):
    def __init__(self, t, fix_names=False):
        # Local imports to get rid of tflite dependency for non-tflite models
        from tflite.TensorType import TensorType as TType

        name = t.Name().decode()
        shape = tuple([t.Shape(si) for si in range(0, t.ShapeLength())])

        type_lookup = {
            TType.FLOAT32: "float32",
            TType.UINT8: "uint8",
            TType.INT8: "int8",
            TType.INT32: "int32",
            TType.BOOL: "int8",
            TType.INT64: "int64",
        }
        dtype = type_lookup[t.Type()]
        super().__init__(name, shape, dtype, fix_names=fix_names)


class RelayTensorInfo(TensorInfo):
    def __init__(self, name, shape, dtype, fix_names=False):
        super().__init__(name, shape, dtype, fix_names=fix_names)


class ModelInfo:
    def __init__(self, in_tensors, out_tensors, fix_names=False):
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors

    def validate(self):
        assert len(self.in_tensors) > 0, "Missing inputs"
        assert len(self.out_tensors) > 0, "Missing outputs"

    @property
    def has_ins(self):
        return len(self.in_tensors) > 0

    @property
    def has_outs(self):
        return len(self.out_tensors) > 0


class TfLiteModelInfo(ModelInfo):
    def __init__(self, model, fix_names=False):
        assert model.SubgraphsLength() == 1
        g = model.Subgraphs(0)

        in_tensors = []
        for i in range(0, g.InputsLength()):
            t = g.Tensors(g.Inputs(i))
            in_tensors.append(TfLiteTensorInfo(t, fix_names=fix_names))

        out_tensors = []
        for i in range(0, g.OutputsLength()):
            t = g.Tensors(g.Outputs(i))
            out_tensors.append(TfLiteTensorInfo(t, fix_names=fix_names))
        super().__init__(in_tensors, out_tensors)


def shape_from_str(shape_str):
    return tuple(
        [complex(*map(int, x.split("i", 1))) if "i" in x else int(x) for x in shape_str.replace(" ", "").split(",")]
    )


def parse_relay_main(line):
    input_tensors = []
    output_tensors = []

    input_tensors_strs = re.compile(r"%[a-zA-Z0-9_]+\s?: Tensor\[\((?:\d+)(?:,\s*\d+)*\), (?:[a-zA-Z0-9_]+)\]").findall(
        line
    )
    for input_tensors_str in input_tensors_strs:
        res = re.compile(r"%([a-zA-Z0-9_]+)\s?: Tensor\[\(([\di]+(?:, [\di]+)*)\), ([a-zA-Z0-9_]+)\]").match(
            input_tensors_str
        )
        assert res is not None
        groups = res.groups()
        assert len(groups) == 3
        input_name, input_shape_str, input_type = groups
        if "v_param" in input_name:
            continue
        input_shape = shape_from_str(input_shape_str)
        input_tensor = TensorInfo(input_name, input_shape, input_type)
        input_tensors.append(input_tensor)

    output_tensor_names_str = re.compile(r"output_tensor_names=\[(\".*\")\]").findall(line)

    output_tensors_str = re.compile(r"-> (.+) {").findall(line)
    # The following depends on InferType annocations
    if len(output_tensors_str) > 0:
        output_tensor_strs = re.compile(r"Tensor\[\([\di]+(?:, [\di]+)*\), [a-zA-Z0-9_]+\]|(?:u?int\d+)").findall(
            output_tensors_str[0]
        )

        if len(output_tensor_names_str) > 0:
            output_tensor_names = re.compile(r"\"([a-zA-Z0-9_]+)\"").findall(output_tensor_names_str[0])
        else:
            output_tensor_names = [f"output{i}" for i in range(len(output_tensor_strs))]

        assert len(output_tensor_names) == len(output_tensor_strs)

        for i, output_name in enumerate(output_tensor_names):
            res = re.compile(r"Tensor\[\(([\di]+(?:, [\di]+)*)\), ([a-zA-Z0-9_]+)\]").match(output_tensor_strs[i])
            if res is None:
                res = re.compile(r"(u?int\d+)").match(output_tensor_strs[i])
            assert res is not None
            groups = res.groups()
            assert len(groups) in [1, 2]
            if len(groups) == 2:
                output_shape_str, output_type = groups
            elif len(groups) == 1:
                output_shape_str, output_type = "1, 1", groups[0]
            output_shape = shape_from_str(output_shape_str)
            output_tensor = TensorInfo(output_name, output_shape, output_type)
            output_tensors.append(output_tensor)
    return input_tensors, output_tensors


class RelayModelInfo(ModelInfo):
    def __init__(self, mod_text, fix_names=False):
        in_tensors = None
        out_tensors = None
        for line in mod_text.split("\n"):
            if "def @main(" in line:
                in_tensors, out_tensors = parse_relay_main(line)
                break
        assert (
            in_tensors is not None and len(in_tensors) > 0
        ), "RelayModelInfo: input_tensors not found (Add TypeInfer details or provided types/shapes manually)"
        assert (
            out_tensors is not None and len(out_tensors) > 0
        ), "RelayModelInfo: output tensors not found (Add TypeInfer details or provide types/shapes manually)"
        super().__init__(in_tensors, out_tensors)


def get_tfgraph_inout(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if op.type == "Placeholder":
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)
    return (inputs, outputs)


class PBModelInfo(ModelInfo):
    def __init__(self, model_file):
        import tensorflow as tf

        with tf.io.gfile.GFile(model_file, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)
            inputs, outputs = get_tfgraph_inout(graph)
        in_tensors = [TensorInfo(t.name, t.shape.as_list(), t.dtype.name) for op in inputs for t in op.outputs]
        out_tensors = [TensorInfo(t.name, t.shape.as_list(), t.dtype.name) for op in outputs for t in op.outputs]
        super().__init__(in_tensors, out_tensors)


class ONNXModelInfo(ModelInfo):
    def __init__(self, model_file):
        from google.protobuf.json_format import MessageToDict
        import onnx
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

        model = onnx.load(model_file)

        def _helper(tensors):
            ret = []
            for tensor in tensors:
                d = MessageToDict(tensor)
                name = d["name"]
                tensor_type = d["type"]["tensorType"]
                elem_type = tensor_type["elemType"]
                dims = tensor_type["shape"]["dim"]
                shape = [int(x["dimValue"]) if "dimValue" in x else 40 for x in dims]  # TODO: dyn shape
                dtype = str(TENSOR_TYPE_TO_NP_TYPE[elem_type])
                ret.append(TensorInfo(name, shape, dtype))
            return ret

        in_tensors = _helper(model.graph.input)
        out_tensors = _helper(model.graph.output)

        # TVM seems to ignore the original output names for ONNX models
        if len(out_tensors) == 1:
            out_tensors[0].name = "output"
        else:
            for i, t in enumerate(out_tensors):
                t.name = f"output{i}"
        super().__init__(in_tensors, out_tensors)


class PaddleModelInfo(ModelInfo):
    def __init__(self, model_file):
        import paddle

        paddle.enable_static()
        paddle.disable_signal_handler()

        exe = paddle.static.Executor(paddle.CPUPlace())
        prefix = "".join(model_file.strip().split(".")[:-1])
        program, _, _ = paddle.static.load_inference_model(prefix, exe)

        output_names = list()
        for block in program.blocks:
            for op in block.ops:
                if op.type == "fetch":
                    output_names.append(op.input("X")[0])

        def _convert_dtype_value(val):
            """Converts a Paddle type id to a string."""
            # See: https://github.com/apache/tvm/blob/cc769fdc951707b8be991949864817b955a4dbc7/python/tvm/
            # relay/frontend/paddlepaddle.py#L70

            convert_dtype_map = {
                21: "int8",
                20: "uint8",
                6: "float64",
                5: "float32",
                4: "float16",
                3: "int64",
                2: "int32",
                1: "int16",
                0: "bool",
            }
            if val not in convert_dtype_map:
                msg = "Paddle data type value %d is not handled yet." % (val)
                raise NotImplementedError(msg)
            return convert_dtype_map[val]

        raise NotImplementedError
        # super().__init__(in_tensors, out_tensors)


def get_tflite_model_info(model_buf):
    # Local imports to get rid of tflite dependency for non-tflite models
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(model_buf, 0)
    model_info = TfLiteModelInfo(tflite_model)
    return model_info


def get_relay_model_info(mod_text):
    model_info = RelayModelInfo(mod_text)
    return model_info


def get_pb_model_info(model_file):
    model_info = PBModelInfo(model_file)
    return model_info


def get_paddle_model_info(model_file):
    model_info = PaddleModelInfo(model_file)
    return model_info


def get_onnx_model_info(model_file):
    model_info = ONNXModelInfo(model_file)
    return model_info


def get_model_info(model, backend_name="unknown"):
    ext = os.path.splitext(model)[1][1:]
    fmt = ModelFormats.from_extension(ext)
    if fmt == ModelFormats.TFLITE:
        with open(model, "rb") as handle:
            model_buf = handle.read()
            return "tflite", get_tflite_model_info(model_buf)
    elif fmt == ModelFormats.RELAY:
        # Warning: the wrapper generateion does currently not work because of the
        # missing possibility to get the relay models input names and shapes
        with open(model, "r") as handle:
            mod_text = handle.read()
        return "relay", get_relay_model_info(mod_text)
    elif fmt == ModelFormats.PB:
        return "pb", get_pb_model_info(model)
    elif fmt == ModelFormats.ONNX:
        return "onnx", get_onnx_model_info(model)
    elif fmt == ModelFormats.PADDLE:
        return "pdmodel", get_pb_model_info(model)
    else:
        raise RuntimeError(f"Unsupported model format '{fmt.name}' for backend '{backend_name}'")


def get_fallback_model_info(model, input_shapes, output_shapes, input_types, output_types, backend_name="unknown"):
    ext = os.path.splitext(model)[1][1:]
    fmt = ModelFormats.from_extension(ext)

    def helper(shapes, types):
        return [TensorInfo(name, shape, types[name]) for name, shape in shapes.items()]

    info = ModelInfo(
        in_tensors=helper(input_shapes, input_types),
        out_tensors=helper(output_shapes, output_types),
    )

    if fmt == ModelFormats.TFLITE:
        return "tflite", info
    elif fmt == ModelFormats.RELAY:
        return "relay", info
    elif fmt == ModelFormats.PB:
        return "pb", info
    elif fmt == ModelFormats.ONNX:
        return "onnx", info
    elif fmt == ModelFormats.PADDLE:
        return "pdmodel", info
    else:
        raise RuntimeError(f"Unsupported model format '{fmt.name}' for backend '{backend_name}'")


def get_model_format(model):
    ext = os.path.splitext(model)[1][1:]
    fmt = ModelFormats.from_extension(ext)

    if fmt:
        return fmt.extension
    else:
        return ext


def get_supported_formats():
    return [ModelFormats.TFLITE, ModelFormats.RELAY, ModelFormats.PB, ModelFormats.ONNX, ModelFormats.PADDLE]
