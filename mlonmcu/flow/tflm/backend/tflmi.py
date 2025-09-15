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
import sys
from typing import Tuple

from .backend import TFLMBackend
from mlonmcu.config import str2bool, str2list, str2dict
from mlonmcu.flow.backend import main
from mlonmcu.artifact import Artifact, ArtifactFormat


# TODO: move to another place
def make_hex_array(data):
    out = ""
    for x in data:
        out += "0x{:02x}, ".format(x)
    return out


class TFLMICodegen:
    def __init__(self):
        pass

    def makeCustomOpPrototypes(self, custom_ops):
        out = "namespace tflite {\n"
        for op in custom_ops:
            op_name = op
            op_reg = op
            if "|" in op:
                op_name, op_reg = op.split("|")[:2]
            out += "extern TfLiteRegistration *" + op_reg + "(void);\n"
        out += "}  //namespace tflite\n"
        return out

    def make_op_registrations(self, ops, custom_ops, reporter=True):
        out = (
            "static tflite::MicroMutableOpResolver<" + str(len(ops) + len(custom_ops)) + "> resolver(error_reporter);\n"
            if reporter
            else "static tflite::MicroMutableOpResolver<" + str(len(ops) + len(custom_ops)) + "> resolver;\n"
        )
        for op in ops:
            if reporter:
                out += (
                    "  if (resolver.Add"
                    + op
                    + '() != kTfLiteOk) {\n    error_reporter->Report("Add'  # TODO: replace with new logger
                    + op
                    + '() failed");\n    return 1;\n  }\n'
                )
            else:
                # TODO
                out += "  if (resolver.Add" + op + "() != kTfLiteOk) {\n    return 1;\n  }\n"
        for op in custom_ops:
            op_name = op
            op_reg = op
            if "|" in op:
                op_name, op_reg = op.split("|")[:2]
            if reporter:
                out += (
                    '  if (resolver.AddCustom("'
                    + op_name
                    + '", tflite::'
                    + op_reg
                    + '()) != kTfLiteOk) {\n    error_reporter->Report("AddCustom'  # TODO: replace with new logger
                    + op_name
                    + '() failed");\n    return 1;\n  }\n'
                )
            else:
                # TODO
                out += (
                    '  if (resolver.AddCustom("'
                    + op_name
                    + '", tflite::'
                    + op_reg
                    + "()) != kTfLiteOk) {\n    return 1;\n  }\n"
                )
        return out

    def generate_header(self, prefix="model"):
        upper_prefix = prefix.upper()
        code = f"""
// This file is generated. Do not edit.
#ifndef {upper_prefix}_GEN_H
#define {upper_prefix}_GEN_H

#include <stddef.h>

int {prefix}_init();
void *{prefix}_input_ptr(int index);
size_t {prefix}_input_size(int index);
size_t {prefix}_inputs();
int {prefix}_invoke();
void *{prefix}_output_ptr(int index);
size_t {prefix}_output_size(int index);
size_t {prefix}_outputs();

#endif  // {upper_prefix}_GEN_H
"""
        return code

    def generate_wrapper(
        self,
        model,
        prefix="model",
        header=True,
        legacy=False,
        debug_arena=False,
        arena_size=None,
        ops=None,
        custom_ops=None,  # TODO: implement
        registrations=None,  # TODO: implement
        ops_resolver=None,  # TODO: implement
        reporter=True,
    ):
        arena_size = arena_size if arena_size is not None else TFLMIBackend.DEFAULTS["arena_size"]

        def convert_op_name(op):
            new_name = ""
            prev = None
            for i, c in enumerate(op):
                if c == "_":
                    prev = c
                    continue
                elif prev is None:
                    new_name += c
                elif (prev == "_") or prev.isdigit():
                    new_name += c
                elif prev.isupper():
                    new_name += c.lower()
                elif c.islower():
                    new_name += c
                prev = c
            # workarounds for strange op names
            MAPPINGS = {
                "Lstm": "LSTM",
            }
            for key, value in MAPPINGS.items():
                new_name = new_name.replace(key, value)
            return new_name

        if len(ops) > 0:
            op_names = list(map(convert_op_name, ops))

            ops = op_names
        if len(custom_ops) > 0:
            raise NotImplementedError
        if len(registrations) > 0:
            raise NotImplementedError
        if ops_resolver == "mutable":
            assert (
                len(ops) > 0
            ), "No ops specified for ops_resolver=mutable!Set model ops in definition.yml or use ops_resolver=all"
        elif ops_resolver == "all":
            raise RuntimeError("AllOpsResolver was removed from TFLM!Use ops_resolver=mutable or ops_resolver=fallback")
        elif ops_resolver == "fallback":
            ops_resolver = "mutable"
            # Defines common operators which are used in many models
            default_ops = [
                "ADD",
                "AVERAGE_POOL_2D",
                "CONCATENATION",
                "CONV_2D",
                "DEPTHWISE_CONV_2D",
                "FULLY_CONNECTED",
                # "GATHER",
                # "LOGISTIC",
                "MAX_POOL_2D",
                # "MEAN",
                # "REDUCE_MAX",
                "RESHAPE",
                "SOFTMAX",
            ]
            default_ops = list(map(convert_op_name, default_ops))
            ops = list(set(ops + default_ops))  # remove duplicates
        else:
            raise ValueError(f"Unsupported ops_resolver: {ops_resolver}")

        model_data = None
        with open(model, "rb") as model_buf:
            model_data = model_buf.read()

        if header:
            header_content = self.generate_header()

        wrapper_content = """
// This file is generated. Do not edit.
#include "printing.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
"""
        if reporter:
            wrapper_content += """#include "tensorflow/lite/micro/micro_error_reporter.h"
"""
        if legacy:
            wrapper_content += """#include "tensorflow/lite/version.h"
"""
        wrapper_content += """
#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif
"""
        if debug_arena:  # This will enable the feature only if it is not overwritten by the user/compiler
            wrapper_content += """
#ifndef DEBUG_ARENA_USAGE
#define DEBUG_ARENA_USAGE 1
#endif

"""
        wrapper_content += """const unsigned char g_model_data[] ALIGN(16) = { """
        wrapper_content += make_hex_array(model_data)
        wrapper_content += """ };

"""
        wrapper_content += self.makeCustomOpPrototypes(custom_ops)
        wrapper_content += """

namespace {
"""
        if reporter:
            wrapper_content += """
tflite::ErrorReporter *error_reporter = nullptr;
"""
        wrapper_content += """
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = """
        wrapper_content += (
            str(arena_size)
            + """;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);

class DummyReporter : public tflite::ErrorReporter {
public:
  ~DummyReporter() {}
  int Report(const char *, va_list) override { return 0; }

private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};
} // namespace

// The name of this function is important for Arduino compatibility.
"""
            + f"int {prefix}_init() {{"
            + """
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
"""
        )
        if reporter:
            wrapper_content += """
#ifdef _DEBUG
  static tflite::MicroErrorReporter micro_error_reporter;
#else
  static DummyReporter micro_error_reporter;
#endif
  error_reporter = &micro_error_reporter;
"""
        wrapper_content += """
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
"""
        if reporter:
            wrapper_content += """
    error_reporter->Report("Model provided is schema version %d not equal "
                           "to supported version %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
"""
        wrapper_content += """
    return 1;
  }
"""

        if ops_resolver == "mutable":
            wrapper_content += self.make_op_registrations(ops, custom_ops, reporter=reporter)
        elif ops_resolver == "all":
            wrapper_content += """
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::ops::micro::AllOpsResolver resolver;
"""

        if reporter:
            wrapper_content += """

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
"""
        else:
            wrapper_content += """

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
"""
        wrapper_content += """
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
"""
        if reporter:
            wrapper_content += """
    error_reporter->Report("AllocateTensors() failed");
"""
        wrapper_content += """
    return 1;
  }
  return 0;
}
"""

        wrapper_content += f"""
void *{prefix}_input_ptr(int id) {{
  return interpreter->input(id)->data.data;
}}

size_t {prefix}_input_size(int id) {{
  return interpreter->input(id)->bytes;
}}

size_t {prefix}_inputs() {{
  return interpreter->inputs_size();
}}

void *{prefix}_output_ptr(int id) {{
  return interpreter->output(id)->data.data;
}}

size_t {prefix}_output_size(int id) {{
  return interpreter->output(id)->bytes;
}}

size_t {prefix}_outputs() {{
  return interpreter->outputs_size();
}}

int {prefix}_invoke() {{
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {{
"""
        if reporter:
            # TODO
            wrapper_content += """
    error_reporter->Report("Invoke failed\\n");
"""
        wrapper_content += """
    return 1;
  }
#if DEBUG_ARENA_USAGE
  size_t used = interpreter->arena_used_bytes();
"""
        if reporter:
            # TODO
            wrapper_content += """
  error_reporter->Report("Arena Usage after model invocation: %d bytes\\n", used);
"""
        else:
            wrapper_content += """
  // printf("Arena Usage after model invocation: %d bytes\\n", used);
  // printf("# Arena Usage: %d\\n", used);
  mlonmcu_printf("# Arena Usage: %d\\n", used);
  // MicroPrintf("# Arena Usage: %d\\n", used);

"""
        wrapper_content += """
#endif  // DEBUG_ARENA_USAGE
  return 0;
}
"""
        if header:
            return wrapper_content, header_content
        else:
            return wrapper_content


class TFLMIBackend(TFLMBackend):
    name = "tflmi"

    FEATURES = TFLMBackend.FEATURES | {"debug_arena"}

    DEFAULTS = {
        **TFLMBackend.DEFAULTS,
        "arena_size": 2**20,  # 1 MB
        "debug_arena": False,
        "ops": [],
        "custom_ops": [],
        "registrations": {},
        "ops_resolver": "mutable",
        "legacy": False,
        "reporter": False,  # Has to be disabled for support with latest upstream
    }

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)
        self.codegen = TFLMICodegen()
        self.model_data = None
        self.prefix = "model"  # Without the _
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        # TODO: decide if artifacts should be handled by code (str) or file path or binary data

    @property
    def legacy(self):
        value = self.config["legacy"]
        return str2bool(value)

    @property
    def debug_arena(self):
        value = self.config["debug_arena"]
        return str2bool(value)

    @property
    def arena_size(self):
        return int(self.config["arena_size"])

    @property
    def ops(self):
        value = self.config["ops"]
        return str2list(value)

    @property
    def custom_ops(self):
        value = self.config["custom_ops"]
        return str2list(value)

    @property
    def registrations(self):
        value = self.config["registrations"]
        return str2dict(value)

    @property
    def ops_resolver(self):
        return self.config["ops_resolver"]

    @property
    def reporter(self):
        value = self.config["reporter"]
        return str2bool(value)

    def generate(self) -> Tuple[dict, dict]:
        artifacts = []
        assert self.model is not None
        wrapper_code, header_code = self.codegen.generate_wrapper(
            self.model,
            prefix=self.prefix,
            header=True,
            arena_size=self.arena_size,
            debug_arena=self.debug_arena,
            ops=self.ops,
            custom_ops=self.custom_ops,
            registrations=self.registrations,
            ops_resolver=self.ops_resolver,
            legacy=self.legacy,
            reporter=self.reporter,
        )
        artifacts.append(Artifact(f"{self.prefix}.cc", content=wrapper_code, fmt=ArtifactFormat.SOURCE))
        artifacts.append(
            Artifact(
                f"{self.prefix}.cc.h",
                content=header_code,
                fmt=ArtifactFormat.SOURCE,
                optional=False,
            )
        )
        workspace_size_artifact = Artifact(
            "tflmi_arena_size.txt", content=f"{self.arena_size}", fmt=ArtifactFormat.TEXT
        )
        artifacts.append(workspace_size_artifact)
        # TODO: stdout_artifact (Would need to invoke TFLMI in subprocess to get stdout)
        return {"default": artifacts}, {}


if __name__ == "__main__":
    sys.exit(
        main(
            TFLMIBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
# TODO: pin defaults to backend class? -> how to defien features and config?
