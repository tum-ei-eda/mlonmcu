#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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
"""Example extension: adapt an external quantizer to an MLonMCU run stage.

Copy this into your environment's plugins/extensions.py and enable
vars.allow_extensions. Supply the command as an argv list in stage config.
The quantizer and its calibration data are supplied by the user.
"""

from pathlib import Path
import subprocess
import time

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.session.stage import Stage, StageResult, register_stage


@register_stage("quantize", override=True)
class QuantizeStage(Stage):
    def validate(self, context):
        command = context.config.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
            raise ValueError("quantize.command must be a nonempty argv list")
        if set(context.inputs) != {"model"}:
            raise ValueError("QUANTIZE requires a model input binding")
        output_name = context.config.get("output_name", "quantized.tflite")
        if Path(output_name).name != output_name:
            raise ValueError("output_name must be a filename")

    def execute(self, context):
        artifacts, parents, metrics = {}, {}, {}
        source = context.spec["inputs"]["model"].split(".", 1)[0]
        for variant, models in context.inputs["model"].items():
            if len(models) != 1:
                raise ValueError("Expected one model artifact per input variant")
            directory = context.workdir / "outputs" / variant
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / context.config.get("output_name", "quantized.tflite")
            command = [arg.format(input=models[0].path, output=output) for arg in context.config["command"]]
            started = time.monotonic()
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=context.config.get("timeout", 600)
            )
            if not output.is_file():
                raise RuntimeError(f"Quantizer did not produce {output}")
            artifacts[variant] = [
                Artifact(output.name, path=output, fmt=ArtifactFormat.PATH, flags=["model"]),
                Artifact("quantize.log", content=result.stdout + result.stderr, fmt=ArtifactFormat.TEXT),
            ]
            parents[variant] = (source, variant)
            metrics[variant] = {
                "Quantize time [s]": time.monotonic() - started,
                "Quantized model size [bytes]": output.stat().st_size,
            }
        return StageResult(artifacts, parents, metrics)
