import tempfile
from pathlib import Path
import os
from enum import IntEnum


class RunStage(IntEnum):
    NOP = 0
    LOAD = 1  # unimplemented
    BUILD = 2
    COMPILE = 3
    RUN = 4
    POSTPROCESS = 5
    DONE = 6


class Run:
    def __init__(
        self,
        idx=None,
        model=None,
        backend=None,
        target=None,
        features=[],
        cfg={},
        num=1,
        archived=False,
        session=None,
    ):
        self.idx = idx
        self.model = model
        self.backend = backend
        self.artifacts = {}
        self.num = num
        self.archived = archived
        self.session = session
        self.stage = RunStage.NOP
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
        self.cfg = cfg
        self.features = features
        self.result = None

    def __repr__(self):
        probs = []
        if self.model:
            probs.append(str(self.model))
        if self.backend:
            probs.append(str(self.backend))
        if self.target:
            probs.append(str(self.target))
        if self.num:
            probs.append(str(self.num))
        if self.features and len(self.features) > 0:
            probs.append(str(self.features))
        if self.cfg and len(self.cfg) > 0:
            probs.append(str(self.cfg))
        return "Run(" + ",".join(probs) + ")"

    def run(self, context=None):
        # print("RUN RUN:", self)
        pass

    def compile(self, context=None):
        print("COMPILE RUN:", self)
        if context:
            if context.cache:
                mlif_build_dir = context.cache.find_best_match(
                    "mlif_build_dir", flags=[self.backend.shortname, self.target.name]
                )
                print("mlif_build_dir", mlif_build_dir)
        pass

    def build(self, context=None):
        self.backend.load(model=self.artifacts["model"])
        code = self.backend.generate_code()
        self.artifacts["code"] = code

    def process(self, until=RunStage.RUN, context=None):
        # print("PROCESS START")
        if until == RunStage.DONE:
            until = RunStage.DONE - 1
        for stage in range(until):
            stage_funcs = {
                # RunStage.NOP: lambda *args, **kwargs: print("NOP RUN"),
                RunStage.NOP: lambda *args, **kwargs: None,
                # RunStage.LOAD: lambda *args, **kwargs: print("LOAD RUN"),
                RunStage.LOAD: lambda *args, **kwargs: None,
                RunStage.BUILD: self.build,
                RunStage.COMPILE: self.compile,
                RunStage.RUN: self.run,
                RunStage.POSTPROCESS: lambda *args, **kwargs: print("POSTPROCESS RUN"),
            }
            func = stage_funcs[stage]
            if func:
                func(context=context)
            self.stage = stage  # FIXME: The stage_func should update the stage intead?
        # print("PROCESS DONE")

    def export(self):
        pass
        # TODO: write run.txt
        # TODO: export model


# TODO: implement close()? and use closing contextlib?
