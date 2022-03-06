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

from .backend import TFLiteBackend
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

    def make_op_registrations(self, ops, custom_ops):
        out = (
            "static tflite::MicroMutableOpResolver<" + str(len(ops) + len(custom_ops)) + "> resolver(error_reporter);\n"
        )
        for op in ops:
            out += "  if (resolver.Add" + op + "() != kTfLiteOk) {\n    return;\n  }\n"
        for op in custom_ops:
            op_name = op
            op_reg = op
            if "|" in op:
                op_name, op_reg = op.split("|")[:2]
            out += (
                '  if (resolver.AddCustom("'
                + op_name
                + '", tflite::'
                + op_reg
                + "()) != kTfLiteOk) {\n    return;\n  }\n"
            )
        return out

    def generate_header(self, prefix="model"):
        upper_prefix = prefix.upper()
        code = f"""
// This file is generated. Do not edit.
#ifndef {upper_prefix}_GEN_H
#define {upper_prefix}_GEN_H

#include <stddef.h>

void {prefix}_init();
void *{prefix}_input_ptr(int index);
size_t {prefix}_input_size(int index);
size_t {prefix}_inputs();
void {prefix}_invoke();
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
        custom_ops=None,
        registrations=None,
        ops_resolver=None,
    ):

        arena_size = arena_size if arena_size is not None else TFLMIBackend.DEFAULTS["arena_size"]
        ops = ops if ops else TFLMIBackend.DEFAULTS["ops"]
        if len(ops) > 0:

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
                return new_name

            op_names = list(map(convert_op_name, ops))
            ops = op_names
        custom_ops = custom_ops if custom_ops else TFLMIBackend.DEFAULTS["custom_ops"]
        if len(custom_ops) > 0:
            raise NotImplementedError
        registrations = (
            registrations if registrations else TFLMIBackend.DEFAULTS["registrations"]
        )  # TODO: Dict or list?
        if len(registrations) > 0:
            raise NotImplementedError
        ops_resolver = ops_resolver if ops_resolver else TFLMIBackend.DEFAULTS["ops_resolver"]
        if ops_resolver != "mutable":
            raise NotImplementedError

        model_data = None
        with open(model, "rb") as model_buf:
            model_data = model_buf.read()

        code = ""

        if header:
            header_content = self.generate_header()

        wrapper_content = """
// This file is generated. Do not edit.
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
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
tflite::ErrorReporter *error_reporter = nullptr;
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
            + f"void {prefix}_init() {{"
            + """
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
#ifdef _DEBUG
  static tflite::MicroErrorReporter micro_error_reporter;
#else
  static DummyReporter micro_error_reporter;
#endif
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal "
                           "to supported version %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  // static tflite::ops::micro::AllOpsResolver resolver;

"""
        )
        wrapper_content += self.make_op_registrations(ops, custom_ops)
        wrapper_content += """

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }
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

void {prefix}_invoke() {{
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {{
    error_reporter->Report("Invoke failed\\n");
    return;
  }}
#if DEBUG_ARENA_USAGE
  size_t used = interpreter->arena_used_bytes();
  error_reporter->Report("Arena Usage after model invocation: %d bytes\\n", used);
#endif  // DEBUG_ARENA_USAGE
}}
"""
        if header:
            return wrapper_content, header_content
        else:
            return wrapper_content


class TFLMIBackend(TFLiteBackend):

    name = "tflmi"

    FEATURES = ["debug_arena"]

    DEFAULTS = {
        **TFLiteBackend.DEFAULTS,
        "arena_size": 2 ** 16,
        "ops": [],
        "custom_ops": [],
        "registrations": {},
        "ops_resolver": "mutable",
        "legacy": False,
    }

    REQUIRED = TFLiteBackend.REQUIRED + []

    def __init__(self, features=None, config=None, context=None):
        super().__init__(features=features, config=config, context=context)
        self.codegen = TFLMICodegen()
        self.model_data = None
        self.prefix = "model"  # Without the _
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        # TODO: decide if artifacts should be handled by code (str) or file path or binary data

    @property
    def legacy(self):
        return bool(self.config["legacy"])

    def generate_code(self, verbose=False):
        artifacts = []
        assert self.model is not None
        config_map = {key.split(".")[-1]: value for key, value in self.config.items()}
        debug_arena = "debug_arena" in self.features
        wrapper_code, header_code = self.codegen.generate_wrapper(
            self.model,
            prefix=self.prefix,
            header=True,
            debug_arena=debug_arena,
            **config_map,
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
        self.artifacts = artifacts


if __name__ == "__main__":
    sys.exit(
        main(
            TFLMIBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
# TODO: pin defaults to backend class? -> how to defien features and config?
