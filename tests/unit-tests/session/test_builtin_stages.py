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
"""Behavioral coverage for the registered default stages and compatibility API."""

from types import SimpleNamespace

import pytest

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.models.model import Model
from mlonmcu.platform.platform import CompilePlatform, TargetPlatform, TunePlatform
from mlonmcu.session.postprocess.postprocess import RunPostprocess
from mlonmcu.session.run import Run, RunStage
from mlonmcu.session.schedule import SessionScheduler
from mlonmcu.session.stage import Pipeline, STAGES, Stage, StageResult, register_stage, stage_key


def metric(implementation, name, value):
    return Artifact(f"{implementation}_metrics.csv", content=f"{name}\n{value}\n", fmt=ArtifactFormat.TEXT)


@pytest.fixture
def components(tmp_path):
    calls = SimpleNamespace(loaded=[], models=[], records=[], compiled=[], executed=[], postprocessed=[], metrics=[])
    bodies = [b"model"]

    class Frontend:
        name = "frontend"
        config = {}
        DEFAULTS = {}

        def generate_artifacts(self, model):
            calls.loaded.append(model.name)
            return [
                Artifact("original.tflite", raw=bodies.pop(0), fmt=ArtifactFormat.RAW, flags=["model"]),
                metric("load", "Input size", 5),
            ]

        def process_metadata(self, model, cfg):
            cfg.update({"backend.arena_size": 128, "original.input_shapes": {"input": [99]}})
            return []

    class Backend:
        name = "backend"
        DEFAULTS = {}
        needs_target = False

        def __init__(self):
            self.config = {}
            self.model = None

        def reconfigure(self):
            pass

        def load_model(self, model, **metadata):
            calls.models.append((model.read_bytes(), metadata))
            self.model = model

        def generate_artifacts(self):
            metrics = metric("build", "Build size", len(self.model.read_bytes()))
            calls.metrics.append(metrics)
            return [
                Artifact("generated.c", content=self.model.read_bytes().decode(), fmt=ArtifactFormat.SOURCE),
                metrics,
            ]

        def set_tuning_records(self, path, tuner_name):
            calls.records.append((path.read_text(), tuner_name))

        def add_platform_defs(self, name, definitions):
            pass

    class Platform(CompilePlatform, TunePlatform, TargetPlatform):
        def _tune_model(self, model_path, backend, target):
            assert model_path.read_bytes() == b"model"
            from mlonmcu.target.metrics import Metrics

            # Exercise branching from one model to multiple tuning candidates.
            return {
                name: [Artifact(f"{name}.records", content=name, fmt=ArtifactFormat.TEXT, flags=["records", "test"])]
                for name in ("fast", "small")
            }, {name: Metrics() for name in ("fast", "small")}

        def generate_artifacts(self, directory, target):
            content = (directory / "generated.c").read_bytes()
            calls.compiled.append(content)
            return [
                Artifact("firmware.elf", raw=content, fmt=ArtifactFormat.RAW, flags=["executable"]),
                metric("compile", "ROM", len(content)),
            ]

    class Processor(RunPostprocess):
        def post_run(self, report, artifacts):
            calls.postprocessed.append([artifact.name for artifact in artifacts])
            report.main_df["Processed"] = True
            return [Artifact("checked.txt", content="ok", fmt=ArtifactFormat.TEXT)]

    platform = Platform("device")
    target = SimpleNamespace(
        name="target",
        platform=platform,
        config={},
        DEFAULTS={},
        reconfigure=lambda: None,
        add_backend_config=lambda *args, **kwargs: None,
    )

    def execute(path):
        content = path.read_bytes()
        calls.executed.append(content)
        return [metric("run", "Cycles", len(content) * 2)]

    target.generate_artifacts = execute
    model_path = tmp_path / "original.tflite"
    model_path.write_bytes(b"model")
    model = Model("original", [model_path], config={"original.input_shapes": {"input": [1]}})
    return (
        dict(
            model=model,
            frontends=[Frontend()],
            backend=Backend(),
            platforms=[platform],
            target=target,
            framework=SimpleNamespace(name="framework", config={}, DEFAULTS={}),
            postprocesses=[Processor("check")],
        ),
        calls,
        bodies,
    )


def make_run(directory, kwargs, **options):
    directory.mkdir(exist_ok=True)
    run = Run(**kwargs, **options)
    run.dir = directory
    return run


@pytest.mark.parametrize("tuning", [False, True])
@pytest.mark.parametrize("subdirs", [False, True])
def test_default_workflow_through_registered_stages(tmp_path, components, tuning, subdirs):
    kwargs, calls, _ = components
    run = make_run(
        tmp_path, kwargs, config={"run.tune_enabled": tuning, "run.stage_subdirs": subdirs, "run.profile_stages": True}
    )
    report = run.process(until=RunStage.DONE, export=True)
    assert not run.failing, run.reason
    assert run.next_stage == RunStage.DONE
    assert run.backend.config["arena_size"] == 128
    assert run.model.input_shapes == {"input": [1]}  # Metadata must preserve user shapes.
    assert calls.compiled == [b"model"] * (2 if tuning else 1)
    assert calls.executed == calls.compiled
    assert len(calls.records) == (2 if tuning else 0)
    assert all(metadata["input_shapes"] == {"input": [1]} for _, metadata in calls.models)
    assert all(report.df["Processed"])
    assert all(report.df["ROM"] == 5)
    assert all(report.df["Cycles"] == 10)
    assert (RunStage.TUNE in run.completed_stages) == tuning
    assert len(run.times) == len(run.completed_stages)
    assert not run.locked
    assert (tmp_path / "pipeline.yml").is_file()


def test_renamed_default_stages_and_legacy_endpoint_resolution(tmp_path, components):
    kwargs, calls, _ = components
    pipeline = [
        {"id": f"my_{name}", "uses": name} for name in ("load", "tune", "build", "compile", "run", "postprocess")
    ]
    run = make_run(tmp_path, kwargs, pipeline=pipeline)
    run.process(until=RunStage.LOAD)
    assert run.completed[stage_key("my_load")]
    # An explicitly required tune does not depend on the default tune_enabled switch.
    report = run.process(until=RunStage.DONE)
    assert not run.failing, run.reason
    assert len(calls.records) == 2
    assert run.pipeline.resolve(RunStage.RUN) == stage_key("my_run")
    assert all(report.df["Cycles"] == 10)
    assert all(report.df["Build size"] == 5)
    assert (tmp_path / "stages/my_build/sub/fast/my_build_metrics.csv").is_file()
    assert all(artifact.name == "build_metrics.csv" for artifact in calls.metrics)
    assert run.postprocess_stage == stage_key("my_postprocess")
    assert all(report.df["Processed"])


def test_repeated_builds_compile_and_run_use_explicit_producers(tmp_path, components):
    kwargs, calls, bodies = components
    bodies[:] = [b"one", b"two"]
    pipeline = [
        {"id": "load_a", "uses": "load"},
        {"id": "build_a", "uses": "build"},
        {"id": "load_b", "uses": "load"},
        {"id": "build_b", "uses": "build"},
        {"id": "compile_a", "uses": "compile", "inputs": {"code": "build_a.*"}},
        {"id": "run_a", "uses": "run", "inputs": {"program": "compile_a.executable"}},
        {"id": "compile_b", "uses": "compile", "inputs": {"code": "build_b.*"}},
        {"id": "run_b", "uses": "run", "inputs": {"program": "compile_b.executable"}},
    ]
    run = make_run(tmp_path, kwargs, pipeline=pipeline)
    report = run.process(until="done")
    assert not run.failing, run.reason
    assert calls.compiled == [b"one", b"two"]
    assert calls.executed == calls.compiled
    assert run.sub_parents[(stage_key("compile_a"), "default")] == (stage_key("build_a"), "default")
    assert report.df["Cycles"].iloc[0] == 6
    with pytest.raises(ValueError, match="ambiguous"):
        run.pipeline.resolve(RunStage.BUILD)


def test_optional_stages_and_required_capability_errors(tmp_path):
    default = make_run(tmp_path, {})
    assert default.next_stage == RunStage.DONE
    required = make_run(tmp_path, {}, pipeline=[{"id": "required_compile", "uses": "compile"}])
    required.process(until="done")
    assert required.failed_stage == "REQUIRED_COMPILE"
    assert "capabilities" in str(required.reason)
    assert not required.completed[stage_key("required_compile")]
    optional = make_run(tmp_path, {}, pipeline=[{"id": "optional_compile", "uses": "compile", "optional": True}])
    optional.process(until="done")
    assert not optional.failing
    assert optional.next_stage == RunStage.DONE


def test_required_inputs_and_registry_override_share_failure_handling(tmp_path, components, monkeypatch):
    kwargs, _, _ = components
    run = make_run(tmp_path, kwargs, pipeline=["load", "build", "compile"])
    run.process(until="build", skip=["load"])
    assert run.failed_stage == "BUILD"
    assert "incomplete" in str(run.reason)
    assert not run.locked
    assert not run.artifacts_per_stage

    original = STAGES["build"]
    monkeypatch.setitem(STAGES, "build", original)

    @register_stage("build", override=True)
    class Replacement(Stage):
        def execute(self, context):
            assert context.run.locked
            return StageResult({"default": ["invalid artifact"]})

    replacement = make_run(tmp_path, {}, pipeline=["build"])
    replacement.process(until="done")
    assert replacement.failed_stage == "BUILD"
    assert not replacement.completed[RunStage.BUILD]
    assert not replacement.artifacts_per_stage
    assert not replacement.locked


def test_compatibility_method_uses_registered_handler(tmp_path, components):
    kwargs, calls, _ = components
    run = make_run(
        tmp_path, kwargs, pipeline=[{"id": "import_model", "uses": "load"}, {"id": "generate", "uses": "build"}]
    )
    result = run.load()
    assert isinstance(result, StageResult)
    assert calls.loaded == ["original"]
    result = run.build()
    assert isinstance(result, StageResult)
    assert run.completed[stage_key("generate")]
    assert (tmp_path / "stages/generate/generated.c").read_text() == "model"


@pytest.mark.parametrize("per_stage", [False, True])
def test_renamed_postprocessors_handle_partial_session_runs(tmp_path, components, per_stage):
    kwargs, calls, bodies = components
    bodies[:] = [b"first", b"second"]
    runs = []
    for index in range(2):
        pipeline = [
            {"id": f"load_{index}", "uses": "load"},
            {"id": f"build_{index}", "uses": "build"},
            {"id": f"post_{index}", "uses": "postprocess"},
        ]
        run = make_run(tmp_path / str(index), kwargs, idx=index, pipeline=pipeline)
        runs.append(run)
    scheduler = SessionScheduler(runs, until=RunStage.LOAD, per_stage=per_stage, num_workers=1, runs_dir=tmp_path)
    _, results = scheduler.process(context=SimpleNamespace(get_read_only_context=lambda: None))
    assert scheduler.num_failures == 0
    assert len(calls.postprocessed) == 2
    assert all(result.report.df["Processed"].iloc[0] for result in results)
    assert all(run.completed[run.postprocess_stage] for run in runs)
    assert all(not run.completed[stage_key(f"build_{index}")] for index, run in enumerate(runs))


def test_alias_validation_and_registration_contract():
    assert {"load", "tune", "build", "compile", "run", "postprocess", "flash"} <= STAGES.keys()
    assert Pipeline([{"id": "import_model", "uses": "load"}]).resolve(RunStage.LOAD) == stage_key("import_model")
    with pytest.raises(ValueError, match="final"):
        Pipeline([{"id": "report", "uses": "postprocess"}, "load"])
    with pytest.raises(ValueError, match="Unsupported inputs"):
        Pipeline(["load", {"id": "compiler", "uses": "compile", "inputs": {"model": "load.model"}}])
    with pytest.raises(TypeError, match="inherit Stage"):
        register_stage("invalid_class")(object)


@pytest.mark.parametrize("subdirs", [False, True])
def test_program_frontend_compiles_without_backend(tmp_path, components, subdirs):
    from mlonmcu.models.model import Program

    kwargs, calls, _ = components
    kwargs.update(model=Program("program"), backend=None, framework=None)
    kwargs["frontends"][0].generate_artifacts = lambda model: [
        Artifact("generated.c", content="program", fmt=ArtifactFormat.SOURCE)
    ]
    run = make_run(tmp_path, kwargs, config={"run.stage_subdirs": subdirs})
    run.process(until="done")
    assert not run.failing, run.reason
    assert calls.compiled == [b"program"]
    assert calls.executed == [b"program"]
    assert not run.completed[RunStage.BUILD]
    assert run.sub_parents[(RunStage.COMPILE, "default")] == (RunStage.LOAD, "default")


def test_target_runs_build_output_when_compilation_does_not_apply(tmp_path, components):
    kwargs, calls, _ = components
    platform = TargetPlatform("executor")
    kwargs["platforms"] = [platform]
    kwargs["target"].platform = platform
    run = make_run(tmp_path, kwargs)
    run.process(until="done")
    assert not run.failing, run.reason
    assert not run.completed[RunStage.COMPILE]
    assert calls.executed == [b"model"]
    assert run.sub_parents[(RunStage.RUN, "default")] == (RunStage.BUILD, "default")


def test_explicit_tuner_binding_selects_its_model_not_the_latest_tuner(tmp_path, components):
    kwargs, _, bodies = components
    bodies[:] = [b"model", b"model"]
    pipeline = [
        {"id": "load_a", "uses": "load"},
        {"id": "tune_a", "uses": "tune"},
        {"id": "load_b", "uses": "load"},
        {"id": "tune_b", "uses": "tune"},
        {"id": "build_a", "uses": "build", "inputs": {"records": "TUNE_A.*"}},
    ]
    run = make_run(tmp_path, kwargs, pipeline=pipeline)
    run.process(until="done")
    assert not run.failing, run.reason
    spec = run.pipeline.stage_spec(run, stage_key("build_a"))
    assert spec["inputs"]["model"] == "load_a.model"
    assert run.sub_parents[(stage_key("build_a"), "fast")] == (stage_key("tune_a"), "fast")


def test_builtin_stage_cannot_consume_a_skipped_required_producer(tmp_path, components):
    kwargs, _, _ = components
    run = make_run(tmp_path, kwargs, pipeline=["load", "build", "compile"])
    run.process(until="compile", skip=["build"])
    assert run.failed_stage == "COMPILE"
    assert "incomplete" in str(run.reason)
    assert not run.completed[RunStage.COMPILE]
    assert not run.locked


def test_direct_load_initializes_its_working_directory(components):
    kwargs, calls, _ = components
    kwargs["platforms"] = []
    kwargs["target"] = None
    run = Run(**kwargs)
    assert run.dir is None
    result = run.load()
    assert isinstance(result, StageResult)
    assert run.completed[RunStage.LOAD]
    assert calls.loaded == ["original"]
    assert (run.dir / "original.tflite").read_bytes() == b"model"
    run.tempdir.cleanup()
