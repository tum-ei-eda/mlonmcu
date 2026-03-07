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

import numpy as np

from mlonmcu.models.model import ModelFormats


def parse_mlir_signature(mlir_text):

    # Find the util.func @main signature
    match1 = re.search(r"util\.func\s+.*?@(\w+)\(([^)(]*?)\)\s*->\s*\(([^)(]*?)\)\s*\{", mlir_text, re.DOTALL)
    if not match1:
        # match2 = re.search(r"func\.func\s+@(\w+)\(([^)(]*)\)\s*->\s*(([^)(]*))\s+{", mlir_text, re.DOTALL)
        match2 = re.search(
            r"func\.func\s+@([a-zA-Z0-9_\-\"]+)\(([^)(]*?)\)\s*->\s*\(?([^)(]*?)\)?\s*(?:attributes(.*))?\{",
            mlir_text,
            re.DOTALL,
        )
        if not match2:
            raise ValueError("No util.func @main(...) -> (...) { } found.")
        func_name = match2.group(1)
        input_args = match2.group(2)
        output_args = match2.group(3)
        # attr_args = match2.group(4)
    else:

        func_name = match1.group(1)
        input_args = match1.group(2)
        output_args = match1.group(3)
        # attr_args = ""

    inputs = []
    outputs = []
    # print("func_name", func_name)
    # print("input_args", input_args)
    # print("output_args", output_args)
    # print("attr_args", attr_args)

    # Parse inputs
    if input_args.strip():
        for arg in input_args.split(", "):
            parts = arg.strip().split(":", maxsplit=1)
            if len(parts) < 2:
                continue
            arg_name = parts[0].strip()
            tensor_info = parts[1].strip()

            shape_dtype_match = re.search(r"[v]?tensor<([^>]+)>", tensor_info)
            identifier_match = re.search(r'ml_program.identifier\s*=\s*"([^"]+)"', tensor_info)

            if shape_dtype_match:
                shape_dtype_str = shape_dtype_match.group(1)
                if shape_dtype_str.count("x") == 0:  # vtensor
                    assert shape_dtype_str[0] == "["
                    shape_dtype_str = shape_dtype_str[1:]
                    shape_dtype_parts = shape_dtype_str.split("],", 1)
                    shape_parts, dtype = shape_dtype_parts
                    shape_parts = shape_parts.split(",")
                else:
                    shape_dtype_parts = shape_dtype_str.split("x")
                    *shape_parts, dtype = shape_dtype_parts
                shape = [None if dim == "?" else int(dim) for dim in shape_parts]
            else:
                shape = []
                dtype = None

            inputs.append(
                {
                    "arg": arg_name,
                    "shape": shape,
                    "dtype": dtype,
                    "name": identifier_match.group(1) if identifier_match else None,
                }
            )

    # Parse outputs
    if output_args.strip():
        for out in output_args.split("},"):
            out = out.strip()
            shape_dtype_match = re.search(r"[v]?tensor<([^>]+)>", out)
            identifier_match = re.search(r'ml_program.identifier\s*=\s*"([^"]+)"', out)

            if shape_dtype_match:
                shape_dtype_str = shape_dtype_match.group(1)
                if shape_dtype_str.count("x") == 0:  # vtensor
                    assert shape_dtype_str[0] == "["
                    shape_dtype_str = shape_dtype_str[1:]
                    shape_dtype_parts = shape_dtype_str.split("],", 1)
                    shape_parts, dtype = shape_dtype_parts
                    shape_parts = shape_parts.split(",")
                else:
                    shape_dtype_parts = shape_dtype_str.split("x")
                    *shape_parts, dtype = shape_dtype_parts
                shape = [None if dim == "?" else int(dim) for dim in shape_parts]
            else:
                shape = []
                dtype = None

            outputs.append(
                {"shape": shape, "dtype": dtype, "name": identifier_match.group(1) if identifier_match else None}
            )

    return func_name, inputs, outputs


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
        if not isinstance(dtype, str):
            dtype = np.dtype(dtype).name
        assert isinstance(dtype, str)
        assert dtype in size_lookup, f"Unsupported type: {dtype}"
        self.dtype = dtype
        self.type_size = size_lookup[self.dtype]
        # TODO: support dynamic shapes?

    def __repr__(self):
        return f"TensorInfo({self.name}, {self.shape}, {self.dtype}, size={self.size})"

    @property
    def size(self):
        ret = self.type_size
        for dim in self.shape:
            if dim is None:  # assume 1
                continue
            if isinstance(dim, complex):
                real = dim.real
                imag = dim.imag
                assert real == int(real)
                assert imag == int(imag)
                ret *= int(real) + int(imag)
            else:
                ret *= dim
        return ret

    @property
    def c_type(self):
        if self.dtype == "float32":
            return "float"
        elif self.dtype == "uint8":
            return "uint8_t"
        elif self.dtype == "int8":
            return "int8_t"
        elif self.dtype == "uint64":
            return "uint64_t"
        elif self.dtype == "int64":
            return "int64_t"
        else:
            raise RuntimeError(f"Unknown c_type for type: {self.dtype}")


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
    def __init__(self, in_tensors, out_tensors, main_func_name=None, fix_names=False):
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors
        self.main_func_name = main_func_name

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
            # output_tensor_names = [f"output{i}" for i in range(len(output_tensor_strs))]
            output_tensor_names = [f"output{i}" if i > 0 else "output" for i in range(len(output_tensor_strs))]

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

        # from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
        from onnx.helper import tensor_dtype_to_np_dtype

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
                # dtype = str(TENSOR_TYPE_TO_NP_TYPE[elem_type])
                dtype = str(tensor_dtype_to_np_dtype(elem_type))
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


class TorchModelInfo(ModelInfo):
    def __init__(self, model_file):
        from mlonmcu.models.torch_models.torch_utils import load_torch_model

        _, exported, _ = load_torch_model(model_file)

        graph = exported.graph

        import torch

        TORCH_TO_NUMPY_DTYPE = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            # torch.bfloat16: np.dtype("bfloat16"),  # requires numpy >= 1.24
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
            torch.complex64: np.complex64,
            torch.complex128: np.complex128,
        }

        def torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
            if dtype not in TORCH_TO_NUMPY_DTYPE:
                raise ValueError(f"Unsupported torch dtype: {dtype}")
            return TORCH_TO_NUMPY_DTYPE[dtype]

        in_tensors = []
        out_tensors = []
        for node in graph.nodes:
            if node.op == "placeholder":
                meta = node.meta["val"]
                name = node.name
                shape = list(meta.shape)
                dtype = meta.dtype
                np_dtype = torch_dtype_to_numpy(dtype)
                in_tensor = TensorInfo(name, shape, np_dtype)
                in_tensors.append(in_tensor)

            elif node.op == "output":
                for out in node.args[0]:
                    meta = out.meta["val"]
                    name = out.name
                    shape = list(meta.shape)
                    dtype = meta.dtype
                    np_dtype = torch_dtype_to_numpy(dtype)
                    out_tensor = TensorInfo(name, shape, np_dtype)
                    out_tensors.append(out_tensor)

        super().__init__(in_tensors, out_tensors)


class PTEModelInfo(ModelInfo):
    def __init__(self, model_file):
        # import executorch.exir as exir
        from executorch.runtime import Runtime
        from executorch.exir.schema import ScalarType

        # Load the PTE file
        # program = exir.load(model_file)
        runtime = Runtime.get()
        program = runtime.load_program(model_file)
        metadata = program.metadata("forward")
        # method = program.load_method("forward")
        in_tensors = []
        out_tensors = []
        num_inputs = metadata.num_inputs()
        num_outputs = metadata.num_outputs()

        def _helper(meta, name):
            shape = meta.sizes()
            dtype = meta.dtype()

            EXECUTORCH_DTYPE_TO_NUMPY = {
                ScalarType.FLOAT: np.float32,
                ScalarType.DOUBLE: np.float64,
                ScalarType.HALF: np.float16,
                # ScalarType.BFLOAT16: np.dtype("bfloat16"),
                ScalarType.CHAR: np.int8,
                ScalarType.SHORT: np.int16,
                ScalarType.INT: np.int32,
                ScalarType.LONG: np.int64,
                ScalarType.BYTE: np.uint8,
                ScalarType.BOOL: np.bool_,
            }
            numpy_dtype = EXECUTORCH_DTYPE_TO_NUMPY.get(dtype)
            assert numpy_dtype is not None, f"Unhandled PTE dtype: {dtype}"
            # numpy_dtype = np.dtype(numpy_dtype).name
            tensor_info = TensorInfo(name, shape, numpy_dtype)
            return tensor_info

        for i in range(num_inputs):
            name = f"input{i}" if i > 0 else "input"
            input_tensor_meta = metadata.input_tensor_meta(i)
            in_tensor = _helper(input_tensor_meta, name)

            in_tensors.append(in_tensor)

        for i in range(num_outputs):
            name = f"output{i}" if i > 0 else "output"
            output_tensor_meta = metadata.output_tensor_meta(i)
            out_tensor = _helper(output_tensor_meta, name)

            out_tensors.append(out_tensor)

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


class MLIRModelInfo(ModelInfo):
    def __init__(self, mod_text, fix_names=False):
        in_tensors = []
        out_tensors = []
        func_name, inputs, outputs = parse_mlir_signature(mod_text)
        type_lookup = {
            "i8": "int8",
            "i32": "int32",
            "f32": "float32",
        }
        for inp in inputs:
            input_name = inp["name"]
            dtype = inp["dtype"]
            input_type = type_lookup.get(dtype)
            assert input_type is not None, f"Unsupported dtype: {dtype}"
            input_shape = inp["shape"]
            input_tensor = TensorInfo(input_name, input_shape, input_type)
            in_tensors.append(input_tensor)
        for outp in outputs:
            output_name = outp["name"]
            dtype = outp["dtype"]
            output_shape = outp["shape"]
            if dtype is None and output_name is None and len(output_shape) == 0:
                continue
            output_type = type_lookup.get(dtype)
            assert output_type is not None, f"Unsupported dtype: {dtype}"
            output_tensor = TensorInfo(output_name, output_shape, output_type)
            out_tensors.append(output_tensor)

        super().__init__(in_tensors, out_tensors, main_func_name=func_name)


def get_tflite_model_info(model_buf):
    # Local imports to get rid of tflite dependency for non-tflite models
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(model_buf, 0)
    model_info = TfLiteModelInfo(tflite_model)
    return model_info


def get_relay_model_info(mod_text):
    model_info = RelayModelInfo(mod_text)
    return model_info


def get_mlir_model_info(mod_text):
    model_info = MLIRModelInfo(mod_text)
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


def get_pte_model_info(model_file):
    model_info = PTEModelInfo(model_file)
    return model_info


def get_torch_model_info(model_file):
    model_info = TorchModelInfo(model_file)
    return model_info


# def get_torch_python_model_info(model_file):
#     raise NotImplementedError
#     # TODO: read class from python file?
#     return get_torch_model_info(model_class)
#
#
# def get_torch_pickle_model_info(model_file):
#     import pickle
#
#     with open(model_file, "rb") as f:
#         model_class = pickle.load(f)
#     return get_torch_model_info(model_class)


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
    elif fmt == ModelFormats.PTE:
        return "pte", get_pte_model_info(model)
    elif fmt in [ModelFormats.TORCH_PYTHON, ModelFormats.TORCH_PICKLE, ModelFormats.TORCH_EXPORTED]:
        return "torch", get_torch_model_info(model)
    # elif fmt == ModelFormats.TORCH_PYTHON:
    #     return "torch_python", get_torch_model_info(model)
    # elif fmt == ModelFormats.TORCH_PICKLE:
    #     return "torch_pickle", get_torch_model_info(model)
    elif fmt == ModelFormats.MLIR:
        with open(model, "r") as handle:
            mod_text = handle.read()
        return "mlir", get_mlir_model_info(mod_text)
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
    elif fmt == ModelFormats.MLIR:
        return "mlir", info
    elif fmt == ModelFormats.PTE:
        return "pte", info
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


def get_supported_formats_iree():
    return [ModelFormats.TFLITE, ModelFormats.ONNX, ModelFormats.SAVED_MODEL, ModelFormats.MLIR, ModelFormats.PB]
