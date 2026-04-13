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
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import mlonmcu.setup.utils as utils
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.config import str2bool, str2dict
from mlonmcu.flow.backend import main
from mlonmcu.logging import get_logger

from .backend import ONNXBackend

logger = get_logger()


class Onnx2CBackend(ONNXBackend):
    name = "onnx2c"

    FEATURES = set()

    DEFAULTS = {
        "print_outputs": False,
        "log_level": 0,
        "func_name": "entry",
        "no_globals": False,
        "extern_init": False,
        "only_init": False,
        "avr": False,
        "optimizations": None,
        "define": {},
        "sanitize_legacy_broadcast_attrs": True,
        "sanitized_model_out": None,
    }

    REQUIRED = ONNXBackend.REQUIRED | {"onnx2c.exe"}

    @property
    def print_outputs(self):
        return str2bool(self.config["print_outputs"])

    @property
    def log_level(self):
        return int(self.config["log_level"])

    @property
    def func_name(self):
        return self.config["func_name"]

    @property
    def no_globals(self):
        return str2bool(self.config["no_globals"])

    @property
    def extern_init(self):
        return str2bool(self.config["extern_init"])

    @property
    def only_init(self):
        return str2bool(self.config["only_init"])

    @property
    def avr(self):
        return str2bool(self.config["avr"])

    @property
    def optimizations(self):
        return self.config["optimizations"]

    @property
    def define(self):
        value = self.config["define"]
        if isinstance(value, str):
            value = str2dict(value)
        assert isinstance(value, dict)
        return value

    @property
    def sanitize_legacy_broadcast_attrs(self):
        return str2bool(self.config["sanitize_legacy_broadcast_attrs"])

    @property
    def sanitized_model_out(self):
        return self.config["sanitized_model_out"]

    def _sanitize_model_for_onnx2c(self, model_path, out_path):
        if not self.sanitize_legacy_broadcast_attrs:
            return model_path
        try:
            import numpy as np
            import onnx
            from onnx import TensorProto, helper, numpy_helper
        except ImportError:
            logger.warning("ONNX package not available, skipping attribute sanitization for onnx2c")
            return model_path

        model = onnx.load(str(model_path))
        changed = False
        legacy_axis = {}

        initializer_map = {init.name: init for init in model.graph.initializer}

        def upsert_initializer(new_init):
            for i in range(len(model.graph.initializer) - 1, -1, -1):
                if model.graph.initializer[i].name == new_init.name:
                    del model.graph.initializer[i]
            model.graph.initializer.append(new_init)
            initializer_map[new_init.name] = new_init

        def get_shape_map():
            ret = {}
            for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
                tt = vi.type.tensor_type
                if not tt.HasField("shape"):
                    continue
                dims = []
                valid = True
                for dim in tt.shape.dim:
                    if dim.HasField("dim_value"):
                        dims.append(int(dim.dim_value))
                    else:
                        valid = False
                        break
                if valid:
                    ret[vi.name] = dims
            for init in model.graph.initializer:
                ret[init.name] = list(init.dims)
            return ret

        attr_names = {"axis", "broadcast"}
        ops = {"Add", "Sub", "Mul", "Div", "Sum", "Max", "Min", "Mean"}

        for node in model.graph.node:
            if node.op_type not in ops:
                continue
            axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
            if axis_attr is not None and node.name:
                legacy_axis[node.name] = int(axis_attr.i)
            keep = [attr for attr in node.attribute if attr.name not in attr_names]
            if len(keep) != len(node.attribute):
                del node.attribute[:]
                node.attribute.extend(keep)
                changed = True

        nodes_to_drop = set()
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Reshape":
                continue
            shape_attr = None
            keep = []
            for attr in node.attribute:
                if attr.name == "shape":
                    shape_attr = attr
                else:
                    keep.append(attr)
            if shape_attr is None:
                continue

            if len(node.input) >= 1 and node.input[0] in initializer_map:
                shape_vals = list(shape_attr.ints)
                if shape_vals:
                    src_arr = numpy_helper.to_array(initializer_map[node.input[0]])
                    reshaped = np.reshape(src_arr, shape_vals)
                    upsert_initializer(numpy_helper.from_array(reshaped, name=node.output[0]))
                    nodes_to_drop.add(node.name if node.name else f"__idx_{idx}")
                    changed = True
                    continue

            if len(node.input) >= 2:
                del node.attribute[:]
                node.attribute.extend(keep)
                changed = True
                continue

            shape_vals = list(shape_attr.ints)
            if not shape_vals:
                logger.warning("Reshape node has empty legacy shape attribute: %s", node.name)
                continue
            shape_name = f"{node.name or ('reshape_' + str(idx))}_shape"
            upsert_initializer(helper.make_tensor(shape_name, TensorProto.INT64, [len(shape_vals)], shape_vals))
            node.input.append(shape_name)
            del node.attribute[:]
            node.attribute.extend(keep)
            changed = True

        if nodes_to_drop:
            new_nodes = []
            for idx, node in enumerate(model.graph.node):
                key = node.name if node.name else f"__idx_{idx}"
                if key not in nodes_to_drop:
                    new_nodes.append(node)
            del model.graph.node[:]
            model.graph.node.extend(new_nodes)

        shape_map = get_shape_map()
        consumers = {}
        for node in model.graph.node:
            for inp in node.input:
                consumers.setdefault(inp, []).append(node)

        for node in model.graph.node:
            if node.op_type != "Reshape" or len(node.input) < 2 or len(node.output) == 0:
                continue
            if not node.name.endswith("_reshape1"):
                continue
            shape_init_name = node.input[1]
            if shape_init_name not in initializer_map:
                continue
            shape_vals = numpy_helper.to_array(initializer_map[shape_init_name]).tolist()
            if not isinstance(shape_vals, list) or len(shape_vals) != 1:
                continue
            channel = int(shape_vals[0])
            if channel <= 0:
                continue
            for cons in consumers.get(node.output[0], []):
                if cons.op_type not in ops or len(cons.input) < 2:
                    continue
                if cons.op_type == "Add" and cons.name.startswith("Plus"):
                    upsert_initializer(numpy_helper.from_array(np.array([1, channel, 1, 1], dtype=np.int64), name=shape_init_name))
                    changed = True
                    break
                other = cons.input[0] if cons.input[1] == node.output[0] else cons.input[1]
                other_shape = shape_map.get(other)
                if other_shape is not None and len(other_shape) == 4:
                    upsert_initializer(numpy_helper.from_array(np.array([1, channel, 1, 1], dtype=np.int64), name=shape_init_name))
                    changed = True
                    break

        shape_map = get_shape_map()
        producer_by_output = {}
        for node in model.graph.node:
            for output_name in node.output:
                producer_by_output[output_name] = node

        matmulinteger_a_zero_points = set()
        matmulinteger_b_zero_points = set()
        for node in model.graph.node:
            if node.op_type == "MatMulInteger" and len(node.input) >= 3:
                matmulinteger_a_zero_points.add(node.input[2])
            if node.op_type == "MatMulInteger" and len(node.input) >= 4:
                matmulinteger_b_zero_points.add(node.input[3])

        def collapse_uniform_zero_point(name):
            init = initializer_map.get(name)
            if init is None:
                return False
            arr = numpy_helper.to_array(init)
            if arr.size == 0:
                return False
            flat = arr.reshape(-1)
            if not np.all(flat == flat[0]):
                logger.warning(
                    "Collapsing non-uniform MatMulInteger zero-point '%s' to scalar for onnx2c compatibility",
                    name,
                )
            scalar_value = np.array([flat[0]], dtype=arr.dtype)
            upsert_initializer(numpy_helper.from_array(scalar_value, name=name))
            return True

        for name in sorted(matmulinteger_b_zero_points):
            if collapse_uniform_zero_point(name):
                changed = True

        zp_redirects = {}
        rewritten_nodes = []
        for idx, node in enumerate(model.graph.node):
            for input_index, input_name in enumerate(node.input):
                if input_name in zp_redirects:
                    node.input[input_index] = zp_redirects[input_name]

            if node.op_type == "DynamicQuantizeLinear" and len(node.output) >= 3 and node.output[2] in matmulinteger_a_zero_points:
                zp_shape_name = f"{node.name or ('dql_' + str(idx))}_zero_point_shape"
                zp_output_name = f"{node.output[2]}_rank1"
                model.graph.initializer.append(helper.make_tensor(zp_shape_name, TensorProto.INT64, [1], [1]))
                rewritten_nodes.append(node)
                rewritten_nodes.append(
                    helper.make_node(
                        "Reshape",
                        [node.output[2], zp_shape_name],
                        [zp_output_name],
                        name=f"{node.name}_zero_point_fix",
                    )
                )
                zp_redirects[node.output[2]] = zp_output_name
                changed = True
                continue

            if node.op_type not in ops or len(node.input) < 2:
                rewritten_nodes.append(node)
                continue

            axis_attr = None
            keep = []
            for attr in node.attribute:
                if attr.name == "axis":
                    axis_attr = int(attr.i)
                elif attr.name == "broadcast":
                    continue
                else:
                    keep.append(attr)
            if axis_attr is None and node.name in legacy_axis:
                axis_attr = legacy_axis[node.name]
            if axis_attr is None:
                rewritten_nodes.append(node)
                continue

            lhs_shape = shape_map.get(node.input[0])
            rhs_shape = shape_map.get(node.input[1])
            if rhs_shape is None:
                prod = producer_by_output.get(node.input[1])
                if prod is not None and prod.op_type == "Reshape":
                    shape_attr = next((a for a in prod.attribute if a.name == "shape"), None)
                    if shape_attr is not None and len(shape_attr.ints) > 0:
                        rhs_shape = list(shape_attr.ints)
                    elif len(prod.input) >= 2 and prod.input[1] in initializer_map:
                        try:
                            shape_vals = numpy_helper.to_array(initializer_map[prod.input[1]]).tolist()
                            rhs_shape = [int(x) for x in shape_vals]
                        except Exception:
                            rhs_shape = None

            if rhs_shape is None:
                rewritten_nodes.append(node)
                continue

            if lhs_shape is None:
                if axis_attr == 1 and len(rhs_shape) == 1:
                    lhs_shape = [1, int(rhs_shape[0]), 1, 1]
                else:
                    rewritten_nodes.append(node)
                    continue

            lhs_rank = len(lhs_shape)
            rhs_rank = len(rhs_shape)
            if axis_attr < 0:
                axis_attr += lhs_rank
            if rhs_rank > lhs_rank or axis_attr < 0 or axis_attr + rhs_rank > lhs_rank:
                rewritten_nodes.append(node)
                continue

            target_shape = [1] * lhs_rank
            target_shape[axis_attr : axis_attr + rhs_rank] = list(rhs_shape)

            reshape_out = f"{node.input[1]}_axisfix"
            shape_name = f"{node.name or ('node_' + str(idx))}_axis_shape"
            model.graph.initializer.append(helper.make_tensor(shape_name, TensorProto.INT64, [len(target_shape)], target_shape))
            rewritten_nodes.append(
                helper.make_node("Reshape", [node.input[1], shape_name], [reshape_out], name=f"{node.name}_axisfix")
            )
            node.input[1] = reshape_out
            del node.attribute[:]
            node.attribute.extend(keep)
            rewritten_nodes.append(node)
            changed = True

        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)

        if not changed:
            return model_path

        onnx.save(model, str(out_path))
        logger.info("Sanitized legacy ONNX broadcast attributes for onnx2c: %s", out_path)
        return out_path

    def _parse_function_signature(self, source_text, func_name):
        match = re.search(rf"void\s+{re.escape(func_name)}\s*\((.*?)\)\s*\{{", source_text, re.DOTALL)
        if not match:
            raise RuntimeError(f"Unable to locate onnx2c function signature: {func_name}(...) in generated source")
        signature = match.group(1).strip()
        if not signature:
            return []

        args = []
        current = []
        depth = 0
        for char in signature:
            if char == "," and depth == 0:
                arg = "".join(current).strip()
                if arg:
                    args.append(arg)
                current = []
                continue
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            current.append(char)
        final_arg = "".join(current).strip()
        if final_arg:
            args.append(final_arg)
        return args

    def _build_runtime_shim(self, source_text):
        args = self._parse_function_signature(source_text, self.func_name)
        input_args = []
        output_args = []
        arg_decls = []

        for index, arg in enumerate(args):
            match = re.match(r"^(?P<type>.+?)\s+(?P<name>[A-Za-z_]\w*)(?P<dims>(?:\[[^\]]+\])*)\s*$", arg)
            if not match:
                raise RuntimeError(f"Unable to parse onnx2c function argument: {arg}")
            arg_type = match.group("type").replace("const ", "")
            arg_name = match.group("name")
            arg_dims = match.group("dims")
            if not arg_dims:
                raise RuntimeError(f"onnx2c function argument has no array dimensions: {arg}")
            arg_decls.append(f"static {arg_type} {arg_name}{arg_dims};")
            if index == 0:
                input_args.append((arg_name, f"sizeof({arg_name})"))
            else:
                output_args.append((arg_name, f"sizeof({arg_name})"))

        if not input_args:
            raise RuntimeError("onnx2c runtime shim needs at least one model input")
        if not output_args:
            raise RuntimeError("onnx2c runtime shim needs at least one model output")

        call_args = ", ".join([name for name, _ in input_args + output_args])
        input_calls = "\n    ".join([f"ret = mlif_request_input({name}, {size}, &new_);" for name, size in input_args])
        output_calls = "\n              ".join([f"ret = mlif_handle_result({name}, {size});" for name, size in output_args])

        return f"""#include "ml_interface.h"

#include <stdbool.h>
#include <stddef.h>

{chr(10).join(arg_decls)}

extern void {self.func_name}({', '.join(args)});

int mlonmcu_init() {{
  return 0;
}}

int mlonmcu_deinit() {{
  return 0;
}}

int mlonmcu_run() {{
  size_t remaining = NUM_RUNS;
  while (remaining) {{
    {self.func_name}({call_args});
    remaining--;
  }}
  return 0;
}}

int mlonmcu_check() {{
  size_t input_num = 0;
  int ret = 0;
  bool new_ = false;
  while (true) {{
    {input_calls}
    if (ret) {{
      return ret;
    }}
    if (!new_) {{
      break;
    }}
    if (input_num == {len(input_args) - 1}) {{
      {self.func_name}({call_args});
      {output_calls}
      if (ret) {{
        return ret;
      }}
      input_num = 0;
    }} else {{
      input_num++;
    }}
  }}
  return ret;
}}
"""

    def generate(self) -> Tuple[dict, dict]:
        assert self.model is not None
        artifacts = []
        onnx2c_exe = self.config["onnx2c.exe"]

        base_name = Path(self.model).stem

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = Path(self.model)
            sanitized_path = Path(tmpdirname) / f"{base_name}.sanitized.onnx"
            model_to_use = self._sanitize_model_for_onnx2c(model_path, sanitized_path)

            args = []
            if self.avr:
                args.append("--avr")
            if self.no_globals:
                args.append("--no-globals")
            if self.extern_init:
                args.append("--extern-init")
            if self.only_init:
                args.append("--only-init")

            args.extend(["--log", str(self.log_level)])
            args.extend(["--func-name", self.func_name])

            if self.optimizations:
                args.extend(["--optimizations", str(self.optimizations)])

            for dim, size in self.define.items():
                args.extend(["--define", f"{dim}:{size}"])

            args.append(str(model_to_use))

            out = utils.execute(onnx2c_exe, *args, live=self.print_outputs)
            source_artifact = Artifact(f"{base_name}.c", content=out, fmt=ArtifactFormat.SOURCE)
            artifacts.append(source_artifact)

            shim_content = self._build_runtime_shim(out)
            shim_artifact = Artifact(f"{base_name}_mlif.c", content=shim_content, fmt=ArtifactFormat.SOURCE)
            artifacts.append(shim_artifact)

            if model_to_use != model_path:
                with open(model_to_use, "rb") as in_file:
                    sanitized_data = in_file.read()
                out_path = self.sanitized_model_out
                if out_path:
                    out_path = Path(out_path)
                    if not out_path.is_absolute():
                        out_path = Path(os.getcwd()) / out_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(sanitized_data)
                artifacts.append(
                    Artifact(
                        f"{base_name}.sanitized.onnx",
                        raw=sanitized_data,
                        fmt=ArtifactFormat.RAW,
                    )
                )

            artifacts.append(Artifact("onnx2c_out.log", content=out, fmt=ArtifactFormat.TEXT))

        return {"default": artifacts}, {}


if __name__ == "__main__":
    sys.exit(
        main(
            Onnx2CBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
