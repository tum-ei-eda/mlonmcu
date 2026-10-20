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
"""Optional pipeline stages implemented using existing platform capabilities."""

from contextlib import nullcontext
from pathlib import Path
import tempfile

from filelock import FileLock

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.platform.platform import TargetPlatform
from .stage import Stage, StageResult, stage_key


class FlashStage(Stage):
    """Program one variant, keeping its board locked through the following RUN."""

    def validate(self, context):
        platform = context.run.target_platform
        if platform is None or not platform.supports_flash:
            raise ValueError("FLASH requires a platform with flash support")
        if type(platform).run is not TargetPlatform.run:
            raise ValueError("Explicit FLASH currently requires the standard TargetPlatform.run implementation")
        if not any(spec.get("resources", {}).get("exclusive") for spec in context.run.pipeline.specs.values()):
            raise ValueError("FLASH requires an exclusive board resource and runs_per_stage=0")
        if set(context.inputs) != {"program"} or len(context.inputs["program"]) != 1:
            raise ValueError("FLASH requires one program variant; use separate runs for different firmware images")

    def execute(self, context):
        platform = context.run.target_platform
        variant, artifacts = next(iter(context.inputs["program"].items()))
        programs = [
            artifact
            for artifact in artifacts
            if artifact.name.endswith((".elf", ".bin", ".bit")) or "executable" in artifact.flags
        ]
        if len(programs) != 1:
            raise ValueError("FLASH requires exactly one ELF, BIN, BIT, or executable-flagged artifact")
        program = programs[0]
        lock = FileLock(Path(tempfile.gettempdir()) / "mlonmcu_serial.lock") if platform.needs_lock else nullcontext()
        with lock:
            output = platform.flash(program.path, context.run.target, timeout=context.config.get("timeout", 120))
        context.run._flashed_program = program.path
        source = stage_key(context.spec["inputs"]["program"].split(".", 1)[0])
        return StageResult(
            {variant: [Artifact("flash.log", content=str(output or ""), fmt=ArtifactFormat.TEXT)]},
            parents={variant: (source, variant)},
        )


BUILTIN_STAGES = {"flash": FlashStage}
