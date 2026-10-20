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
"""User-defined run stages and serializable linear pipelines.

Register implementations from an MLonMCU extensions.py file so fresh workers
can discover them as well. Pipeline descriptions contain data, never callables.
"""

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import re

from mlonmcu.artifact import lookup_artifacts


class StageId(str):
    @property
    def name(self):
        return str(self)


def stage_key(value):
    from .run import RunStage

    if isinstance(value, int):
        return RunStage(value)
    value = str(value).upper()
    return RunStage.__members__.get(value, StageId(value))


STAGES = {}


def register_stage(name, override=False):
    """Register a Stage class; built-in names and accidental duplicates are errors."""
    name = str(name).lower()

    def decorate(cls):
        from .run import RunStage

        if name.upper() in RunStage.__members__ or (name in STAGES and not override):
            raise ValueError(f"Stage already registered: {name}")
        STAGES[name] = cls
        return cls

    return decorate


class Stage:
    def validate(self, context):
        """Raise an exception if configuration or capabilities are invalid."""

    def execute(self, context):
        raise NotImplementedError


@dataclass
class StageResult:
    # Variants map to artifact lists. Every output must declare its parent when
    # inputs exist, including stages which branch or rename their variants.
    artifacts: dict = field(default_factory=dict)
    parents: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


@dataclass
class StageContext:
    run: object
    spec: dict
    inputs: dict
    workdir: Path

    @property
    def config(self):
        return self.spec.get("config", {})


class Pipeline:
    def __init__(self, description=None):
        from .run import RunStage

        if isinstance(description, dict):
            if "stages" not in description:
                raise ValueError("Pipeline mapping requires a stages list")
            description = description["stages"]
        if description is None:
            description = [stage.name.lower() for stage in RunStage if stage not in (RunStage.NOP, RunStage.DONE)]
        if not isinstance(description, list):
            raise ValueError("Pipeline stages must be a list")
        self.specs = {}
        for item in description:
            spec = deepcopy(item) if isinstance(item, dict) else {"id": item, "uses": item}
            if "id" not in spec or not isinstance(spec["id"], str):
                raise ValueError("Each stage requires a string id")
            spec.setdefault("uses", spec["id"])
            key = stage_key(spec["id"])
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key.name):
                raise ValueError(f"Invalid stage ID: {key.name}")
            if key in self.specs or key in (RunStage.NOP, RunStage.DONE):
                raise ValueError(f"Duplicate or reserved stage ID: {key.name}")
            uses = str(spec["uses"]).lower()
            if uses.upper() in RunStage.__members__:
                if key != stage_key(uses):
                    raise ValueError("Built-in stage IDs must match their implementation")
            elif isinstance(key, RunStage):
                raise ValueError("Custom implementations cannot replace built-in stages")
            spec["id"] = key.name.lower()
            spec["uses"] = uses
            spec.setdefault("inputs", {})
            if not isinstance(spec["inputs"], dict) or not isinstance(spec.get("config", {}), dict):
                raise ValueError("Stage inputs and config must be mappings")
            for binding in spec["inputs"].values():
                if not isinstance(binding, str):
                    raise ValueError("Input bindings must be strings")
                source, separator, flag = binding.partition(".")
                if not separator or not flag or stage_key(source) not in self.specs:
                    raise ValueError(f"Input must reference an earlier stage and artifact flag: {binding}")
            if isinstance(key, RunStage) and spec["inputs"]:
                if key not in (RunStage.TUNE, RunStage.BUILD) or set(spec["inputs"]) != {"model"}:
                    raise ValueError("Built-in input bindings currently support only TUNE/BUILD.model")
                if spec["inputs"]["model"].split(".", 1)[1] != "model":
                    raise ValueError("Built-in model bindings require the model artifact flag")
            if RunStage.POSTPROCESS in self.specs:
                raise ValueError("POSTPROCESS must be the final stage")
            if not isinstance(spec.get("resources", {}), dict):
                raise ValueError("Stage resources must be a mapping")
            resources = spec.get("resources", {}).get("exclusive", [])
            if isinstance(resources, str):
                resources = [resources]
            if not isinstance(resources, list) or not all(isinstance(item, str) and item for item in resources):
                raise ValueError("Exclusive resources must be a list of nonempty names")
            if resources:
                spec["resources"] = {"exclusive": resources}
            self.specs[key] = spec

    @property
    def stages(self):
        return list(self.specs)

    def serialize(self):
        return {"stages": deepcopy(list(self.specs.values()))}

    def resolve(self, value):
        from .run import RunStage

        key = stage_key(value)
        if key not in self.specs and key not in (RunStage.NOP, RunStage.DONE):
            raise ValueError(f"Stage is not in this pipeline: {key.name}")
        return key

    def through(self, until):
        from .run import RunStage

        until = self.resolve(until)
        if until == RunStage.DONE:
            return self.stages
        if until == RunStage.NOP:
            return []
        return self.stages[: self.stages.index(until) + 1]

    def inputs(self, run, key):
        result = {}
        for name, binding in self.specs[key]["inputs"].items():
            source, flag = binding.split(".", 1)
            source = stage_key(source)
            if not run.completed.get(source):
                raise RuntimeError(f"Required input stage is incomplete: {source.name}")
            variants = {}
            for variant, artifacts in run.artifacts_per_stage.get(source, {}).items():
                matches = lookup_artifacts(artifacts, flags=None if flag == "*" else [flag])
                if not matches:
                    raise RuntimeError(f"Missing input {binding} for variant {variant}")
                variants[variant] = matches
            if not variants:
                raise RuntimeError(f"Missing input: {binding}")
            result[name] = variants
        return result
