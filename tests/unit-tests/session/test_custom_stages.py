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
"""Exercise custom pipelines through the real run engine and scheduler."""

import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.session.run import Run, RunInitializer, RunStage
from mlonmcu.session.schedule import SessionScheduler, _used_stages
from mlonmcu.session.stage import Pipeline, Stage, StageResult, STAGES, register_stage, stage_key


def model(name="original.tflite", content=b"original"):
    return Artifact(name, raw=content, fmt=ArtifactFormat.RAW, flags=["model"])


@pytest.fixture
def implementations(monkeypatch):
    calls = []

    class Source(Stage):
        def execute(self, context):
            calls.append(context.spec["id"])
            return StageResult({"default": [model()]}, metrics={"default": {"Source size": 8}})

    class Quantize(Stage):
        def execute(self, context):
            calls.append(context.spec["id"])
            assert context.inputs["model"]["default"][0].raw == b"original"
            return StageResult(
                {"int8": [model("int8.tflite", b"int8")], "int4": [model("int4.tflite", b"int4")]},
                parents={"int8": ("source", "default"), "int4": ("source", "default")},
                metrics={"int8": {"Bits": 8}, "int4": {"Bits": 4}},
            )

    monkeypatch.setitem(STAGES, "test_source", Source)
    monkeypatch.setitem(STAGES, "test_quantize", Quantize)
    return calls


def description():
    return [
        {"id": "source", "uses": "test_source"},
        {"id": "quantize", "uses": "test_quantize", "inputs": {"model": "source.model"}},
    ]


def make_run(tmp_path, pipeline, **kwargs):
    run = Run(pipeline=pipeline, **kwargs)
    run.dir = tmp_path
    return run


def test_transform_variants_metrics_export_and_resume(tmp_path, implementations):
    run = make_run(tmp_path, description(), config={"run.stage_subdirs": True, "run.profile_stages": True})
    run.process(until="source")
    assert run.next_stage == stage_key("quantize")
    report = run.process(until=RunStage.DONE, export=True)
    assert not run.failing
    assert implementations == ["source", "quantize"]
    assert run.next_stage == RunStage.DONE
    assert run.last_stage == stage_key("quantize")
    assert dict(zip(report.df["Sub"], report.df["Bits"])) == {"int8": 8, "int4": 4}
    assert list(report.df["Source size"]) == [8, 8]
    assert (tmp_path / "stages/quantize/sub/int8/int8.tflite").read_bytes() == b"int8"
    run.process(until=RunStage.DONE)
    assert implementations == ["source", "quantize"]
    run.process(until="source", start="source")
    assert not run.completed[stage_key("quantize")]
    assert stage_key("quantize") not in run.artifacts_per_stage


def test_build_consumes_quantized_models(tmp_path, implementations):
    loaded = []
    backend = SimpleNamespace(
        name="test",
        needs_target=False,
        config={},
        DEFAULTS={},
        reconfigure=lambda: None,
        load_model=lambda **kwargs: loaded.append(kwargs["model"].read_bytes()),
        generate_artifacts=lambda: [Artifact("generated.c", content="code", fmt=ArtifactFormat.SOURCE)],
    )
    pipeline = description() + [{"id": "build", "inputs": {"model": "quantize.model"}}]
    run = make_run(
        tmp_path,
        pipeline,
        backend=backend,
        framework=SimpleNamespace(name="test", config={}, DEFAULTS={}),
        model=SimpleNamespace(name="original", config={}, DEFAULTS={}),
    )
    run.process(until=RunStage.DONE)
    assert not run.failing, run.reason
    assert loaded == [b"int8", b"int4"]
    assert run.sub_parents[(RunStage.BUILD, "int8")] == (stage_key("quantize"), "int8")
    assert list(run.get_report().df["Source size"]) == [8, 8]


def test_missing_inputs_and_unknown_implementation_fail_at_stage(tmp_path, implementations):
    run = make_run(tmp_path, description())
    run.process(until="quantize", skip=["source"])
    assert run.failed_stage == "QUANTIZE"
    assert not run.completed[stage_key("quantize")]
    run = make_run(tmp_path, [{"id": "unknown", "uses": "missing"}])
    run.process(until="done")
    assert run.failed_stage == "UNKNOWN"
    assert "Unknown stage implementation" in str(run.reason)


@pytest.mark.parametrize("implementation", ["worker_source", "load"])
def test_initializer_roundtrip_and_fresh_worker(tmp_path, implementation):
    extension = tmp_path / "extensions.py"
    extension.write_text(
        "from mlonmcu.session.stage import Stage, StageResult, register_stage\n"
        f'@register_stage("{implementation}", override={implementation == "load"})\n'
        "class WorkerSource(Stage):\n"
        "    def execute(self, context):\n"
        '        return StageResult({"default": []}, metrics={"default": {"Worker metric": 42}})\n'
    )
    initializer = RunInitializer(pipeline=[{"id": "source", "uses": implementation, "config": {"dtype": "int8"}}])
    path = tmp_path / "initializer.yml"
    initializer.save(path)
    restored = RunInitializer.from_file(path)[0]
    assert restored.pipeline.serialize() == initializer.pipeline.serialize()
    assert pickle.loads(pickle.dumps(restored)).pipeline.serialize() == initializer.pipeline.serialize()
    code = """import sys
from pathlib import Path
from types import SimpleNamespace
from mlonmcu.context.context import MlonMcuContextMinimal
from mlonmcu.session.run import RunInitializer
context = object.__new__(MlonMcuContextMinimal)
context.environment = SimpleNamespace(vars={"allow_extensions": True},
    paths={"plugins": SimpleNamespace(path=Path(sys.argv[1]).parent)})
initializer = RunInitializer.from_file(sys.argv[2])[0]
run = initializer.realize(context=context)
# Repeated jobs must not re-register strict extensions in the same worker.
initializer.realize(context=context)
run.dir = Path(sys.argv[3])
report = run.process(until="done", export=True)
assert not run.failing, run.reason
assert report.df["Worker metric"].iloc[0] == 42
"""
    subprocess.run([sys.executable, "-c", code, str(extension), str(path), str(tmp_path)], check=True)


def test_default_pipeline_and_legacy_endpoints():
    run = Run()
    assert run.pipeline.stages == list(RunStage)[1:-1]
    assert run.next_stage == RunStage.DONE
    assert run.process(until=RunStage.RUN) is not None
    assert run.process(until=RunStage.DONE) is not None


def test_scheduler_merges_compatible_orders_and_rejects_cycles():
    a = Run(pipeline=[{"id": "source", "uses": "test_source"}, {"id": "a", "uses": "custom"}])
    b = Run(pipeline=[{"id": "source", "uses": "test_source"}, {"id": "b", "uses": "custom"}])
    stages = _used_stages([a, b], RunStage.DONE)
    assert stages[0] == stage_key("source")
    assert set(stages[1:]) == {stage_key("a"), stage_key("b")}
    c = Run(pipeline=[{"id": "a", "uses": "custom"}, {"id": "source", "uses": "test_source"}])
    with pytest.raises(ValueError, match="incompatible"):
        _used_stages([a, c], RunStage.DONE)


def test_exclusive_resources_cover_execution_and_release_on_failure(tmp_path, monkeypatch):
    import hashlib
    import tempfile
    from filelock import FileLock

    resource = str(tmp_path / "board")
    lock_path = tmp_path.__class__(tempfile.gettempdir()) / (
        "mlonmcu-resource-" + hashlib.sha256(resource.encode()).hexdigest() + ".lock"
    )
    code = """import sys
from filelock import FileLock, Timeout
try:
    with FileLock(sys.argv[1], timeout=0):
        sys.exit(1)
except Timeout:
    pass
"""

    class CheckLock(Stage):
        def execute(self, context):
            subprocess.run([sys.executable, "-c", code, str(lock_path)], check=True)
            if context.config.get("fail"):
                raise RuntimeError("device failure")
            return StageResult({"default": []})

    monkeypatch.setitem(STAGES, "check_lock", CheckLock)
    pipeline = [
        {"id": "flash", "uses": "check_lock", "resources": {"exclusive": resource}},
        {"id": "monitor", "uses": "check_lock", "config": {"fail": True}},
    ]
    run = make_run(tmp_path, pipeline)
    run.process(until="done")
    assert run.failed_stage == "MONITOR"
    with FileLock(lock_path, timeout=0):
        pass
    with pytest.raises(ValueError, match="runs_per_stage=0"):
        SessionScheduler([run], until=RunStage.DONE, per_stage=True)


@pytest.mark.parametrize(
    "pipeline",
    [
        ["load", "load"],
        ["done"],
        [{"id": "../bad", "uses": "custom"}],
        [{"id": "custom", "inputs": {"model": "future.model"}}],
        ["postprocess", "load"],
        ["load", {"id": "compile", "inputs": {"model": "load.model"}}],
    ],
)
def test_invalid_pipelines(pipeline):
    with pytest.raises(ValueError):
        Pipeline(pipeline)


def test_registry_rejects_builtin_and_duplicate_names(monkeypatch):
    monkeypatch.delitem(STAGES, "test_registry", raising=False)
    register_stage("test_registry")(Stage)
    with pytest.raises(ValueError):
        register_stage("test_registry")(Stage)
    with pytest.raises(ValueError):
        register_stage("load")(Stage)
    monkeypatch.delitem(STAGES, "test_registry")


@pytest.mark.parametrize("per_stage", [False, True])
def test_scheduler_executes_heterogeneous_pipelines(tmp_path, implementations, per_stage):
    runs = []
    for idx, pipeline in enumerate([description(), description()[:1]]):
        directory = tmp_path / str(idx)
        directory.mkdir()
        runs.append(make_run(directory, pipeline, idx=idx))
    scheduler = SessionScheduler(
        runs, until=RunStage.DONE, per_stage=per_stage, num_workers=2, batch_size=2, runs_dir=tmp_path
    )
    context = SimpleNamespace(get_read_only_context=lambda: None)
    _, results = scheduler.process(context=context)
    assert scheduler.num_failures == 0
    assert len(results) == 2
    assert all(result is not None and not result.failing for result in results)
    assert runs[0].completed[stage_key("quantize")]
    assert runs[1].next_stage == RunStage.DONE


def test_generic_cli_endpoint():
    import argparse
    from mlonmcu.cli.execute import get_parser

    parser = argparse.ArgumentParser()
    get_parser(parser.add_subparsers())
    args = parser.parse_args(["execute", "--until", "quantize", "--initializer", "pipeline.yml"])
    assert args.until == "quantize"


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("rebuild", [False, True])
def test_explicit_flash_and_run_do_not_program_twice(tmp_path, split, rebuild, monkeypatch):
    from mlonmcu.platform.platform import CompilePlatform, TargetPlatform

    calls = []

    class DevicePlatform(CompilePlatform, TargetPlatform):
        def flash(self, elf, target, timeout=120):
            calls.append("flash")
            return "programmed"

        def monitor(self, target, timeout=60):
            calls.append("monitor")
            return "finished"

    platform = DevicePlatform("device")
    target = SimpleNamespace(name="device", platform=platform, config={}, DEFAULTS={}, reconfigure=lambda: None)

    def execute(elf):
        assert platform.run(elf, target) == "finished"
        return {"default": [Artifact("run_metrics.csv", content="Cycles\n3\n", fmt=ArtifactFormat.TEXT)]}

    target.generate_artifacts = execute
    pipeline = [
        "compile",
        {
            "id": "flash",
            "uses": "flash",
            "inputs": {"program": "compile.*"},
            "resources": {"exclusive": str(tmp_path / "board")},
        },
        "run",
    ]
    if rebuild:
        pipeline.insert(2, {"id": "compile_b", "uses": "compile"})
    run = make_run(tmp_path, pipeline, target=target, platforms=[platform])
    elf = tmp_path / "firmware.elf"
    elf.write_bytes(b"firmware")

    class FirmwareStage(Stage):
        def execute(self, context):
            if context.key == stage_key("compile_b"):
                # PATH artifacts keep this path when exported. Rebuild in place,
                # retaining the size and timestamp to require content identity.
                stat = elf.stat()
                elf.write_bytes(b"rebuild!")
                os.utime(elf, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            return StageResult({"default": [Artifact(elf.name, path=elf, fmt=ArtifactFormat.PATH)]})

    monkeypatch.setitem(STAGES, "compile", FirmwareStage)
    if split:
        run.process(until="flash")
        assert not run.failing, run.reason
        assert calls == ["flash"]
    run.process(until="done")
    assert not run.failing, run.reason
    # A new invocation must not assume that the board still contains our image.
    assert calls == (["flash", "flash", "monitor"] if split or rebuild else ["flash", "monitor"])
    assert not getattr(platform, "_preflashed", False)


def test_external_quantizer_example(tmp_path, implementations, monkeypatch):
    from pathlib import Path
    from mlonmcu.plugins import process_extensions

    monkeypatch.setitem(STAGES, "quantize", Stage)
    process_extensions(Path(__file__).resolve().parents[3] / "examples/custom_stages/extensions.py")
    quantizer = tmp_path / "quantizer.py"
    quantizer.write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes()[:3])\n"
    )
    pipeline = description()[:1] + [
        {
            "id": "quantize",
            "uses": "quantize",
            "inputs": {"model": "source.model"},
            "config": {"command": [sys.executable, str(quantizer), "{input}", "{output}"]},
        }
    ]
    run = make_run(tmp_path, pipeline)
    report = run.process(until="done", export=True)
    assert not run.failing, run.reason
    assert report.df["Quantized model size [bytes]"].iloc[0] == 3
    assert run.artifacts_per_stage[stage_key("quantize")]["default"][0].path.read_bytes() == b"ori"


@pytest.mark.parametrize("per_stage", [False, True])
def test_postprocessing_partial_runs_and_done(tmp_path, implementations, per_stage):
    from mlonmcu.session.schedule import _process_default

    from mlonmcu.session.postprocess.postprocess import RunPostprocess

    calls = []

    class Counter(RunPostprocess):
        def post_run(self, report, artifacts):
            calls.append("postprocess")
            return []

    run = make_run(tmp_path, description() + ["postprocess"], postprocesses=[Counter("counter")])
    _process_default([run], stage_key("source"), [], False, None, tmp_path, False, False, True)
    assert calls == ["postprocess"]
    run.idx = 0
    scheduler = SessionScheduler([run], until="done", per_stage=per_stage, runs_dir=tmp_path)
    scheduler.process(context=SimpleNamespace(get_read_only_context=lambda: None))
    assert scheduler.num_failures == 0
    assert calls == ["postprocess", "postprocess", "postprocess"]
    assert run.completed[RunStage.POSTPROCESS]
