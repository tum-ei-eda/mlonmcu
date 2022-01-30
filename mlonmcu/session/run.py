import tempfile
from pathlib import Path
import os
import copy
from enum import IntEnum

from mlonmcu.logging import get_logger
from mlonmcu.artifact import ArtifactFormat
from mlonmcu.report import Report  # TODO: move to mlonmcu.session.report
from mlonmcu.target.metrics import Metrics

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

    # FEATURES = []

    DEFAULTS = {
        "export_optionals": False,
    }

    # REQUIRED = []

    @classmethod
    def from_file(cls, path):
        """Restore a run object which was written to the disk."""
        raise NotImplementedError

    def __init__(
        self,
        idx=None,
        model=None,
        framework=None,
        frontend=None,
        backend=None,
        target=None,
        mlif=None,  # TODO: rename
        features=None,  # TODO: All features combined or explicit run-features -> postprocesses?
        config=None,  # TODO: All config combined or explicit run-config?
        num=1,
        archived=False,
        session=None,
        comment="",
    ):
        self.idx = idx
        self.model = model  # Model name, not object?
        self.frontend = frontend  # Single one or all enabled ones?
        self.framework = framework  # ???
        self.backend = backend
        self.mlif = mlif
        self.artifacts_per_stage = {}
        self.num = num
        self.archived = archived
        self.session = session
        self.comment = comment
        self.stage = RunStage.NOP  # max executed stage

        self._init_directory()
        self.target = target
        self.config = Run.DEFAULTS
        self.config.update(config if config else {})
        self.features = features if features else []
        self.result = None
        self.active = False  # TODO: rename to staus with enum?

    def _init_directory(self):
        if self.session is None:
            assert not self.archived
            self.tempdir = tempfile.TemporaryDirectory()
            self.dir = Path(self.tempdir.name)
        else:
            self.tempdir = None
            self.dir = self.session.runs_dir / str(self.idx)
            if not self.dir.is_dir():
                os.mkdir(self.dir)
            # This is not a good idea, but else we would need a mutex/lock on the shared build_dir
            # A solution would be to split up the framework runtime libs from the mlif...
            if self.mlif:
                self.mlif.build_dir = self.dir

    def copy(self):
        new = copy.deepcopy(self)
        if self.session:
            new_idx = self.session.request_run_idx()
            new.idx = new_idx
            self._init_directory()
        return new

    def add_frontend(self, frontend):
        self.frontend = frontend

    def add_backend(self, backend):
        self.backend = backend

    def add_framework(self, framework):
        self.framework = framework

    def add_target(self, target):
        self.target = target

    @property
    def export_optional(self):
        return bool(self.config["export_optional"])

    def __repr__(self):
        probs = []
        if self.model:
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
        if stage in self.artifacts_per_stage:
            artifacts = self.artifacts_per_stage[stage]
            for artifact in artifacts:
                if not artifact.optional or optional:
                    dest = self.dir
                    if subdir:
                        stage_idx = int(stage)
                        dest = dest / f"stage_{stage_idx}"
                    extract = artifact.fmt == ArtifactFormat.MLF
                    artifact.export(self.dir, extract=extract)

    def postprocess(self, context=None):  # TODO: drop context arguments?
        """Run the 'run' using the defined target."""
        logger.debug(self.prefix + "Processing stage POSTPROCESS")
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
        logger.debug(self.prefix + "Processing stage RUN")
        assert not self.active, "Parallel processing of the same run is not allowed"
        self.active = True  # TODO: do we need self.current_stage and self.max_stage?
        # Alternative: drop artifacts of higher stages when re-triggering a lower one?

        assert self.stage >= RunStage.COMPILE
        self.export_stage(RunStage.COMPILE, optional=False)
        elf_artifact = self.artifacts_per_stage[RunStage.COMPILE][0]
        self.target.generate_metrics(elf_artifact.path)
        self.artifacts_per_stage[RunStage.RUN] = self.target.artifacts

        self.stage = max(self.stage, RunStage.RUN)
        self.active = False

    def compile(self, context=None):
        """Compile the target software for the run."""
        logger.debug(self.prefix + "Processing stage COMPILE")
        assert not self.active, "Parallel processing of the same run is not allowed"
        self.active = True
        assert self.stage >= RunStage.BUILD

        self.export_stage(RunStage.BUILD, optional=False)
        codegen_dir = self.dir
        # TODO: MLIF -> self.?
        self.mlif.generate_elf(codegen_dir, num=self.num)
        self.artifacts_per_stage[RunStage.COMPILE] = self.mlif.artifacts

        self.stage = max(self.stage, RunStage.COMPILE)
        self.active = False

    def build(self, context=None):
        """Process the run using the choosen backend."""
        logger.debug(self.prefix + "Processing stage BUILD")
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
        logger.debug(self.prefix + "Processing stage LOAD")
        assert not self.active, "Parallel processing of the same run is not allowed"
        self.active = True
        # assert self.stage >= RunStage.NOP

        self.frontend.generate_models(self.model)
        self.artifacts_per_stage[RunStage.LOAD] = self.frontend.artifacts

        self.stage = max(self.stage, RunStage.LOAD)
        self.active = False

    def process(self, until=RunStage.RUN, export=False, context=None):
        """Process the run until a given stage."""
        if until == RunStage.DONE:
            until = RunStage.DONE - 1
        start = self.stage + 1  # self.stage hold the max finished stage
        if start > RunStage.NOP:
            logger.debug(
                # self.prefix + "Processing run until stage %s: %s",
                self.prefix + "Continuing run from stage %s until stage %s",
                str(RunStage(start).name),
                str(RunStage(until).name),
                # str(self),
            )
        else:
            logger.debug(
                # self.prefix + "Processing run until stage %s: %s",
                self.prefix + "Processing run until stage %s",
                str(RunStage(until).name),
                # str(self),
            )
        for stage in range(start, until + 1):
            stage_funcs = {
                RunStage.NOP: lambda *args, **kwargs: None,  # stage already done as self.stage shows
                RunStage.LOAD: self.load,
                RunStage.BUILD: self.build,
                RunStage.COMPILE: self.compile,
                RunStage.RUN: self.run,
                RunStage.POSTPROCESS: self.postprocess,
            }
            func = stage_funcs[stage]
            if func:
                func(context=context)
            # self.stage = stage  # FIXME: The stage_func should update the stage intead?
        report = self.get_report()
        if export:
            self.export(optional=True)  # TODO: set to flase?
            report_file = Path(self.dir) / "report.csv"
            report.export(report_file)
        return report

    def write_run_file(self):
        logger.debug(self.prefix + "Writing run file")
        filename = self.dir / "run.txt"
        with open(filename, "w") as handle:
            handle.write(str(self))

    @property
    def prefix(self):
        return (
            (
                f"[session-{self.session.idx}] [run-{self.idx}] "
                if self.session
                else f"[run-{self.idx}]"
            )
            if self.idx is not None
            else ""
        )

    def get_report(self):
        # TODO: config or args for stuff like (session id) and run id as well as detailed features and configs
        report = Report()
        pre = {}
        if self.session is not None:
            pre["Session"] = self.session.idx
        if self.idx is not None:
            pre["Run"] = self.idx
        if self.model:
            pre["Model"] = self.model.name
        if self.frontend:
            pre["Frontend"] = self.frontend.name
        if self.framework:
            pre["Framework"] = self.framework.name
        if self.backend:
            pre["Backend"] = self.backend.name
        if self.target:
            pre["Target"] = self.target.name
        post = {}
        post["Features"] = None  # TODO: list(all feature names accumulated)
        post["Config"] = None  # TODO: combine all configs with their proper prefix
        post["Comment"] = self.comment
        # if include_sess_idx:
        #     report.session_id = self.session.idx
        # if ?include_run_idx:
        #     report.run_idx = self.idx
        self.export_stage(RunStage.RUN, optional=False)
        if RunStage.RUN in self.artifacts_per_stage:
            metrics_artifact = self.artifacts_per_stage[RunStage.RUN][0]
            metrics = Metrics.from_csv(metrics_artifact.content)
        else:
            metrics = Metrics()
        report.set(
            [
                {
                    **pre,
                    **metrics.get_data(include_optional=True),
                    **post,
                }
            ]
        )
        return report

    def export(self, path=None, optional=False):
        """Write a run configuration to a disk."""
        logger.debug(self.prefix + "Exporting run to disk")
        # TODO use proper locks instead of boolean and also lock during export!
        assert not self.active, "Can not export an active run"
        for stage in range(self.stage):
            self.export_stage(stage)

        self.write_run_file()


# TODO: implement close()? and use closing contextlib?
