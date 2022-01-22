import tempfile
from pathlib import Path
import os
from enum import IntEnum

from mlonmcu.logging import get_logger

logger = get_logger()


class RunStage(IntEnum):
    """Type describing the stages a run can have."""

    NOP = 0
    LOAD = 1  # unimplemented
    BUILD = 2
    COMPILE = 3
    RUN = 4
    POSTPROCESS = 5
    DONE = 6


class Run:
    """A run is single model/backend/framework/target combination with a given set of features and configs."""

    @classmethod
    def from_file(path):
        """Restore a run object which was written to the disk."""
        raise NotImplementedError

    def __init__(
        self,
        idx=None,
        model_name=None,
        frontend=None,
        backend=None,
        target=None,
        mlif=None,  # TODO: rename
        features=None,  # TODO: All features combined or explicit run-features -> postprocesses?
        config=None,  # TODO: All config combined or explicit run-config?
        num=1,
        archived=False,
        session=None,
    ):
        self.idx = idx
        self.model_name = model_name  # Model name, not object?
        self.frontend = frontend  # Single one or all enabled ones?
        # self.framework = framework  # ???
        self.backend = backend
        self.mlif = mlif
        self.artifacts = []
        self.num = num
        self.archived = archived
        self.session = session
        self.stage = RunStage.NOP  # max executed stage
        if self.session is None:
            assert not self.archived
            self.tempdir = tempfile.TemporaryDirectory()
            self.dir = Path(self.tempdir.name)
        else:
            self.tempdir = None
            self.dir = session.runs_dir / str(idx)
            if not self.dir.is_dir():
                os.mkdir(self.dir)
        self.target = target
        self.config = Run.DEFAULTS
        self.config.update(config if config else {})
        self.features = features if features else []
        self.result = None
        self.active = False  # TODO: rename to staus with enum?

    @property
    def export_optional(self):
        return bool(self.config["export_optional"])

    def __repr__(self):
        probs = []
        if self.model_name:
            probs.append(str(self.model))
        if self.frontend:
            probs.append(str(self.frontend))
        if self.backend:
            probs.append(str(self.backend))
        if self.target:
            probs.append(str(self.target))
        if self.num:
            probs.append(str(self.num))
        if self.features and len(self.features) > 0:
            probs.append(str(self.features))
        if self.config and len(self.config) > 0:
            probs.append(str(self.config))
        return "Run(" + ",".join(probs) + ")"

    def toDict(self):
        raise NotImplementedError  # TODO

    def export_stage(self, stage, optional=False, subdir=False):
        # TODO: per stage subdirs?
        artifacts = self.artifacts_per_stage[stage]
        for artifact in artifacts:
            if not artifact.optional or optional:
                dest = subdir
                if subdir:
                    stage_idx = int(stage)
                    dest = dest / f"stage_{stage_idx}"
                artifact.export(self.dir)

    def postprocess(self, context=None):  # TODO: drop context arguments?
        """Run the 'run' using the defined target."""
        assert not self.active, "Parallel processing of the same run is not allowed"
        # TODO: Extract run metrics (cycles, runtime, memory usage,...)
        self.active = True  # TODO: move to self.set_active(stage), self.set_done(stage), self.set_failed(stage)
        assert (
            self.stage >= RunStage.RUN
        )  # Alternative: allow to trigger previous stages recursively as a fallback

        self.export_stage(RunStage.RUN, optional=False)
        self.result = (
            None  # Artifact(f"metrics.csv", data=df, fmt=ArtifactFormat.DATAFRAME)
        )

        self.stage = max(self.stage, RunStage.POSTPROCESS)
        self.active = False
        raise NotImplementedError

    def run(self, context=None):  # TODO: drop context arguments?
        """Run the 'run' using the defined target."""
        assert not self.active, "Parallel processing of the same run is not allowed"
        self.active = True  # TODO: do we need self.current_stage and self.max_stage?
        # Alternative: drop artifacts of higher stages when re-triggering a lower one?

        assert self.stage >= RunStage.COMPILE
        self.export_stage(RunStage.COMPILE, optional=False)
        elf_artifact = self.artifacts_per_stage[RunStage.COMPILE][0]
        self.target.exec(elf_artifact.path)

        self.stage = max(self.stage, RunStage.RUN)
        self.active = False
        raise NotImplementedError

    def compile(self, context=None):
        """Compile the target software for the run."""
        assert not self.active, "Parallel processing of the same run is not allowed"
        self.active = True
        assert self.stage >= RunStage.BUILD

        self.export_stage(RunStage.BUILD, optional=False)
        codegen_dir = self.dir
        # TODO: MLIF -> self.?
        self.mlif.compile(codegen_dir, num=self.num)

        self.stage = max(self.stage, RunStage.COMPILE)
        self.active = False
        raise NotImplementedError

    def build(self, context=None):
        """Process the run using the choosen backend."""
        assert not self.active, "Parallel processing of the same run is not allowed"
        self.active = True
        assert self.stage >= RunStage.LOAD

        self.export_stage(RunStage.LOAD, optional=False)
        model_artifact = self.artifacts_per_stage[RunStage.LOAD][0]
        if not model_artifact.exported:
            model_artifact.export(self.dir)

        # TODO: allow raw data as well as filepath in backends
        self.backend.load_model(model=model_artifact.path)
        self.backend.generate_code()
        self.artifacts_per_stage[RunStage.BUILD] = self.backend.artifacts

        self.stage = max(self.stage, RunStage.BUILD)
        self.active = False

    def load(self, context=None):
        """Load the model using the given frontend."""
        assert not self.active, "Parallel processing of the same run is not allowed"
        self.active = True
        # assert self.stage >= RunStage.NOP

        models = resolve_models(self.model_name)  # TODO
        self.frontend.generate_artifacts(models)
        self.artifacts_per_stage[RunStage.LOAD] = self.frontend.artifacts

        self.stage = max(self.stage, RunStage.LOAD)
        self.active = False

    def process(self, until=RunStage.RUN, context=None):
        """Process the run until a given stage."""
        logger.debug(
            "Started processing a new run until stage %s: %s",
            str(RunStage(until).name),
            str(self),
        )
        if until == RunStage.DONE:
            until = RunStage.DONE - 1
        for stage in range(until):
            stage_funcs = {
                RunStage.NOP: lambda *args, **kwargs: None,  # stage already done as self.stage shows
                RunStage.LOAD: lambda *args, **kwargs: self.load,
                RunStage.BUILD: self.build,
                RunStage.COMPILE: self.compile,
                RunStage.RUN: self.run,
                RunStage.POSTPROCESS: lambda *args, **kwargs: self.postprocess,
            }
            func = stage_funcs[stage]
            if func:
                func(context=context)
            # self.stage = stage  # FIXME: The stage_func should update the stage intead?

    def write_run_file(self):
        filename = self.dir / "run.txt"
        with open(filename, "w") as handle:
            handle.write(str(self))

    def export(self, path=None, optional=False):
        """Write a run configuration to a disk."""
        # TODO use proper locks instead of boolean and also lock during export!
        assert not self.active, "Can not export an active run"
        for stage in range(self.stage):
            self.export_stage(stage)

        self.write_run_file()


# TODO: implement close()? and use closing contextlib?
