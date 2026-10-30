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
"""Default run stages, registered using the public stage contract."""

from contextlib import nullcontext
from pathlib import Path
import hashlib
import tempfile

from filelock import FileLock

from mlonmcu.artifact import Artifact, ArtifactFormat, lookup_artifacts
from mlonmcu.config import filter_config
from mlonmcu.models.model import Model
from mlonmcu.platform.platform import TargetPlatform
from .postprocess.postprocess import RunPostprocess
from .stage import Stage, StageResult, register_stage, stage_key


def program_identity(path):
    """Identify the current image even when a build reuses its output path."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return (path, digest.digest())


def bind_previous(run, key, name, implementation, flag="*"):
    """Resolve a default input to the nearest enabled implementation instance."""
    source = run.pipeline.find(implementation, before=key, run=run)
    return {name: f"{source.name.lower()}.{flag}"} if source is not None else {}


def add_outputs(result, outputs, parent, variant="default"):
    """Preserve the existing variant naming convention at component boundaries."""
    if not isinstance(outputs, dict):
        outputs = {"default": outputs or []}
    for child, artifacts in outputs.items():
        name = child if variant in ("", "default") else (variant if child in ("", "default") else f"{variant}_{child}")
        if name in result.artifacts:
            raise ValueError(f"Duplicate output variant: {name}")
        result.artifacts[name] = artifacts
        if parent is not None:
            result.parents[name] = parent


def configure_backend_target(run):
    use_target = run.target_to_backend and run.target is not None
    if run.backend.needs_target or run.target_optimized_layouts or run.target_optimized_schedules:
        if run.target is None:
            raise ValueError("Backend requires a target")
        use_target = True
    if use_target:
        run.target.add_backend_config(
            run.backend.name,
            run.backend.config,
            optimized_layouts=run.target_optimized_layouts,
            optimized_schedules=run.target_optimized_schedules,
        )


@register_stage("load")
class LoadStage(Stage):
    input_names = set()

    def available(self, run):
        return run.model is not None and bool(run.frontends)

    def execute(self, context):
        run = context.run
        artifacts = run.frontend.generate_artifacts(run.model)
        cfg_new = {}
        if isinstance(run.model, Model):
            metadata = run.frontend.process_metadata(run.model, cfg=cfg_new)
            if metadata is not None:
                if isinstance(artifacts, dict):
                    for variant_artifacts in artifacts.values():
                        variant_artifacts.extend(metadata)
                else:
                    artifacts.extend(metadata)
            for key, value in cfg_new.items():
                component, name = key.split(".")[:2]
                if run.backend is not None and component == run.backend.name:
                    run.backend.config[name] = value
                elif component == run.model.name:
                    if run.model.config[name] is None:
                        run.model.config = filter_config({key: value}, run.model.name, run.model.config, set(), set())
                else:
                    for platform in run.platforms:
                        if platform is not None and component == platform.name:
                            platform.config[name] = value
                run.config[key] = value
        result = StageResult()
        add_outputs(result, artifacts, None)
        return result


@register_stage("tune")
class TuneStage(Stage):
    input_names = {"model"}

    def available(self, run):
        return run.backend is not None and run.tune_platform is not None

    def enabled(self, run):
        return run.tune_enabled and run.backend is not None

    def default_inputs(self, run, key):
        return bind_previous(run, key, "model", "load", "model")

    def execute(self, context):
        context.require_inputs("model")
        run = context.run
        configure_backend_target(run)
        result = StageResult()
        for name, models in context.inputs["model"].items():
            if len(models) != 1:
                raise ValueError("TUNE requires exactly one model per variant")
            outputs = run.tune_platform.tune_model(models[0].path, run.backend, run.target)
            add_outputs(result, outputs, (context.source("model"), name), name)
        return result


@register_stage("build")
class BuildStage(Stage):
    input_names = {"model", "records"}

    def available(self, run):
        return run.backend is not None and run.framework is not None

    def default_inputs(self, run, key):
        defaults = bind_previous(run, key, "model", "load", "model")
        records = run.pipeline.specs[key]["inputs"].get("records")
        if records is not None:
            candidate = stage_key(records.split(".", 1)[0])
            tune = candidate if run.pipeline.specs[candidate]["uses"] == "tune" else None
        else:
            tune = run.pipeline.find("tune", before=key, run=run)
        if tune is not None:
            defaults["records"] = f"{tune.name.lower()}.*"
            tuner_spec = run.pipeline.stage_spec(run, tune)
            if "model" in tuner_spec["inputs"]:
                defaults["model"] = tuner_spec["inputs"]["model"]
        return defaults

    def execute(self, context):
        context.require_inputs("model")
        run = context.run
        configure_backend_target(run)
        result = StageResult()
        model_source = context.source("model")
        records_source = context.source("records") if "records" in context.inputs else None
        if records_source is not None and run.pipeline.specs[records_source]["uses"] == "tune":
            tuner_model = run.pipeline.stage_spec(run, records_source)["inputs"].get("model")
            if tuner_model is not None:
                tuner_source, tuner_flag = tuner_model.split(".", 1)
                same_model = (
                    context.source("model").name.lower() == tuner_source.lower()
                    and context.spec["inputs"]["model"].split(".", 1)[1] == tuner_flag
                )
            else:
                same_model = False
            if not same_model:
                raise ValueError("BUILD and TUNE must bind the same model source")
        variants = context.inputs.get("records", context.inputs["model"])
        for name, items in variants.items():
            model_name = name
            if records_source is not None:
                records = lookup_artifacts(items, fmt=ArtifactFormat.TEXT, flags=["records"], first_only=True)
                if not records:
                    records = lookup_artifacts(
                        items, fmt=ArtifactFormat.ARCHIVE, flags=["records", "metascheduler"], first_only=True
                    )
                if not records:
                    continue
                record = records[0]
                run.backend.set_tuning_records(record.path, tuner_name=record.flags[-1])
                ancestor = (records_source, name)
                while ancestor[0] is not None and ancestor[0] != model_source:
                    ancestor = run.sub_parents.get(ancestor, (None, None))
                if ancestor[0] == model_source:
                    model_name = ancestor[1]
            models = context.inputs["model"].get(model_name, [])
            if len(models) != 1:
                raise ValueError(f"BUILD requires one model for variant {model_name}")
            model = models[0]
            metadata = dict(
                input_shapes=None, output_shapes=None, input_types=None, output_types=None, params_path=None
            )
            if run.model is not None and model.name.split(".", 1)[0] == run.model.name:
                metadata = {name: getattr(run.model, name, None) for name in metadata}
            run.backend.load_model(model=model.path, **metadata)
            add_outputs(result, run.backend.generate_artifacts(), (records_source or model_source, name), name)
            if run.compile_platform:
                run.backend.add_platform_defs(run.compile_platform.name, run.compile_platform.definitions)
        return result


@register_stage("compile")
class CompileStage(Stage):
    input_names = {"code"}

    def available(self, run):
        return run.target is not None and bool(run.platforms) and run.compile_platform is not None

    def default_inputs(self, run, key):
        return bind_previous(run, key, "code", "build") or (
            bind_previous(run, key, "code", "load") if not isinstance(run.model, Model) else {}
        )

    def execute(self, context):
        context.require_inputs("code")
        run = context.run
        source = context.source("code")
        result = StageResult()
        for name in context.inputs["code"]:
            codegen_dir = run.stage_directory(source, name)
            outputs = run.compile_platform.generate_artifacts(codegen_dir, run.target)
            add_outputs(result, outputs, (source, name), name)
        return result


@register_stage("run")
class RunStage(Stage):
    input_names = {"program", "deployment"}

    def available(self, run):
        return run.target is not None and bool(run.platforms) and run.target_platform is not None

    def default_inputs(self, run, key):
        bindings = bind_previous(run, key, "program", "compile") or bind_previous(run, key, "program", "build")
        flash = run.pipeline.find("flash", before=key, run=run)
        if flash is not None and run.completed.get(flash):
            bindings["deployment"] = f"{flash.name.lower()}.*"
        return bindings

    def execute(self, context):
        context.require_inputs("program")
        run = context.run
        result = StageResult()
        for name, artifacts in context.inputs["program"].items():
            if not artifacts:
                raise ValueError(f"RUN requires a program for variant {name}")
            # Explicit flag bindings can select a program anywhere in the
            # producer's artifacts. Wildcard defaults retain the legacy order.
            program = artifacts[0]
            flashed_program = getattr(run, "_flashed_program", None)
            preflashed = flashed_program is not None and flashed_program == program_identity(program.path)
            if not preflashed:
                run._flashed_program = None
                run._flash_parent = None
            execution = run.target_platform.preflashed() if preflashed else nullcontext()
            with execution:
                outputs = run.target.generate_artifacts(program.path)
            parent = (context.source("program"), name)
            flashed = getattr(run, "_flash_parent", None)
            if preflashed and flashed is not None:
                parent = flashed
            add_outputs(result, outputs, parent, name)
        return result


@register_stage("postprocess")
class PostprocessStage(Stage):
    input_names = {"artifacts"}

    is_postprocess = True

    def available(self, run):
        return bool(run.postprocesses)

    def default_inputs(self, run, key):
        previous = run.pipeline.stages[: run.pipeline.stages.index(key)]
        source = next(
            (stage for stage in reversed(previous) if run.completed.get(stage) and run.artifacts_per_stage.get(stage)),
            None,
        )
        return {"artifacts": f"{source.name.lower()}.*"} if source is not None else {}

    def execute(self, context):
        run = context.run
        report = run.get_report()
        result = StageResult(report=report)
        if "artifacts" not in context.inputs:
            return result
        source = context.source("artifacts")
        for name in context.inputs["artifacts"]:
            before = run.get_all_sub_artifacts(name, source)
            merged = []
            for processor in run.postprocesses:
                if isinstance(processor, RunPostprocess):
                    outputs = processor.post_run(report, before)
                    if outputs is None:
                        continue
                    # Existing postprocessors return lists, but support the
                    # variant mapping accepted by the previous Run method too.
                    if isinstance(outputs, dict):
                        for child, artifacts in outputs.items():
                            child_name = (
                                child
                                if name in ("", "default")
                                else (name if child in ("", "default") else f"{name}_{child}")
                            )
                            result.artifacts.setdefault(child_name, []).extend(artifacts)
                            result.parents[child_name] = (source, name)
                            before.extend(artifacts)
                    else:
                        merged.extend(outputs)
                        before.extend(outputs)
            result.artifacts.setdefault(name, []).extend(merged)
            result.parents[name] = (source, name)
        return result


@register_stage("flash")
class FlashStage(Stage):
    """Program one variant, keeping its board locked through the following RUN."""

    input_names = {"program"}

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
        run = context.run
        platform = run.target_platform
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
        identity = program_identity(program.path)
        run._flashed_program = None
        run._flash_parent = None
        with lock:
            output = platform.flash(program.path, run.target, timeout=context.config.get("timeout", 120))
        run._flashed_program = identity
        run._flash_parent = (context.key, variant)
        return StageResult(
            {variant: [Artifact("flash.log", content=str(output or ""), fmt=ArtifactFormat.TEXT)]},
            parents={variant: (context.source("program"), variant)},
        )


# Optional dependencies are imported only when a transformation executes.
from . import model_stages  # noqa: E402,F401
