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
"""Definition of a MLonMCU Run which represents a single benchmark instance for a given set of options."""
import os
import copy
import tempfile
from pathlib import Path
from enum import IntEnum

from mlonmcu.logging import get_logger
from mlonmcu.artifact import ArtifactFormat, lookup_artifacts
from mlonmcu.platform.platform import CompilePlatform, TargetPlatform
from mlonmcu.report import Report  # TODO: move to mlonmcu.session.report
from mlonmcu.config import resolve_required_config, filter_config
from mlonmcu.models.lookup import lookup_models
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import get_matching_features, get_available_features
from mlonmcu.target.metrics import Metrics
from mlonmcu.models import SUPPORTED_FRONTENDS
from mlonmcu.platform import get_platforms
from mlonmcu.flow import SUPPORTED_FRAMEWORKS, SUPPORTED_BACKENDS

from .postprocess import SUPPORTED_POSTPROCESSES
from .postprocess.postprocess import RunPostprocess

logger = get_logger()


class RunStage(IntEnum):
    """Type describing the stages a run can have."""

    NOP = 0
    LOAD = 1  # unimplemented
    TUNE = 2
    BUILD = 3
    COMPILE = 4
    RUN = 5
    POSTPROCESS = 6
    DONE = 7


class Run:
    """A run is single model/backend/framework/target combination with a given set of features and configs."""

    FEATURES = ["autotune", "target_optimized"]

    DEFAULTS = {
        "export_optional": False,
        "tune_enabled": False,
        "target_to_backend": False,
    }

    REQUIRED = []

    @classmethod
    def from_file(cls, path):
        """Restore a run object which was written to the disk."""
        raise NotImplementedError

    def __init__(
        self,
        idx=None,
        model=None,
        framework=None,
        frontends=None,
        backend=None,
        target=None,
        platforms=None,  # TODO: rename
        features=None,  # TODO: All features combined or explicit run-features -> postprocesses?
        config=None,  # TODO: All config combined or explicit run-config?
        postprocesses=None,
        num=1,
        archived=False,
        session=None,
        comment="",
    ):
        self.idx = idx
        self.model = model  # Model name, not object?
        self.frontends = frontends if frontends is not None else []
        self.framework = framework  # ???
        self.backend = backend
        self.platforms = platforms if platforms is not None else []
        self.artifacts_per_stage = {}
        self.num = num
        self.archived = archived
        self.session = session
        self.postprocesses = postprocesses if postprocesses else []
        self.comment = comment
        # self.stage = RunStage.NOP  # max executed stage
        self.completed = {stage: stage == RunStage.NOP for stage in RunStage}

        self.init_directory()
        self.target = target
        self.cache_hints = []
        self.config = config if config else {}
        self.features = features if features else []
        self.run_config = config if config else {}
        self.run_features = self.process_features(features)
        self.run_config = filter_config(self.run_config, "run", self.DEFAULTS, self.REQUIRED)
        self.result = None
        self.failing = False  # -> RunStatus
        # self.lock = threading.Lock()  # FIXME: use mutex instead of boolean
        self.locked = False
        self.report = None

    def process_features(self, features):
        """Utility which handles postprocess_features."""
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.RUN)
        for feature in features:
            assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
            feature.add_run_config(self.run_config)
        return features

    @property
    def tune_enabled(self):
        """Get tune_enabled property."""
        return bool(self.run_config["tune_enabled"])

    @property
    def target_to_backend(self):
        """Get target_to_backend property."""
        return bool(self.run_config["target_to_backend"])

    @property
    def compile_platform(self):
        """Get platform for compile stage."""
        if self.target is not None and isinstance(self.target.platform, CompilePlatform):
            return self.target.platform
        for platform in self.platforms:
            if isinstance(platform, CompilePlatform):
                return platform
        return None

    @property
    def target_platform(self):  # TODO: rename to run_stage?
        """Get platform for run stage."""
        if self.target:
            assert isinstance(self.target.platform, TargetPlatform)
            return self.target.platform
        for platform in self.platforms:
            if isinstance(platform, TargetPlatform):
                return platform
        return None

    def has_stage(self, stage):
        """Returns true if the given stage is available for this run."""
        if stage == RunStage.NOP:
            return True
        if stage == RunStage.LOAD:
            return self.model is not None and len(self.frontends) > 0
        if stage == RunStage.TUNE:
            return self.tune_enabled and self.backend is not None
        if stage == RunStage.BUILD:
            return self.backend is not None and self.framework is not None
        if stage == RunStage.COMPILE:
            return self.target is not None and len(self.platforms) > 0 and self.compile_platform is not None
        if stage == RunStage.RUN:
            return self.target is not None and len(self.platforms) > 0 and self.target_platform is not None
        if stage == RunStage.POSTPROCESS:
            return len(self.postprocesses) > 0
        if stage == RunStage.DONE:
            return False
        return False  # TODO: Throw error instead?

    @property
    def next_stage(self):
        """Determines the next not yet completed stage. Returns RunStage.DONE if already completed."""
        for stage in RunStage:
            if not self.completed[stage.value] and self.has_stage(stage):
                return stage
        return RunStage.DONE

    def lock(self):
        """Aquire a mutex to lock the current run."""
        # ret = self.lock.acquire(timeout=0)
        ret = not self.locked
        self.locked = True
        if not ret:
            raise RuntimeError("Parallel processing of the same run is not allowed")

    def unlock(self):
        """Release a mutex to unlock the current run."""
        # self.lock.release()
        self.locked = False

    def init_directory(self):
        """Initialize the temporary directory for this run."""
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
            for platform in self.platforms:  # TODO: only do this if needed! (not for every platform)
                platform.init_directory(path=Path(self.dir) / platform.name)

    def copy(self):
        """Create a new run based on this instance."""
        new = copy.deepcopy(self)
        if self.session:
            new_idx = self.session.request_run_idx()
            new.idx = new_idx
            self.init_directory()
        return new

    def init_component(self, component_cls, context=None):
        """Helper function to create and configure a MLonMCU component instance for this run."""
        required_keys = component_cls.REQUIRED
        self.config.update(
            resolve_required_config(
                required_keys,
                features=self.features,
                config=self.config,
                cache=context.cache if context else None,
                hints=self.cache_hints,
            )
        )
        component_config = self.config.copy()  # TODOL get rid of this
        return component_cls(features=self.features, config=component_config)

    def add_model(self, model):
        """Setter for the model instance."""
        self.model = model

    def add_frontend(self, frontend):
        """Setter for the frontend instance."""
        self.frontends = [frontend]

    def add_frontends(self, frontends):
        """Setter for the list of frontends."""
        self.frontends = frontends

    def add_backend(self, backend):
        """Setter for the backend instance."""
        self.backend = backend
        # assert len(self.platforms) > 0, "Add at least a platform before adding a backend."
        if self.model is not None:
            assert self.backend.supports_model(self.model), (
                "The added backend does not support the chosen model."
                "Add the backend before adding a model to find a suitable frontend."
            )
        for platform in self.platforms:
            self.backend.add_platform_defs(platform.name, platform.definitions)

    def add_framework(self, framework):
        """Setter for the framework instance."""
        self.framework = framework
        # assert len(self.platforms) > 0, "Add at least a platform before adding a framework."
        for platform in self.platforms:
            self.framework.add_platform_defs(platform.name, platform.definitions)

    def add_target(self, target):
        """Setter for the target instance."""
        self.target = target
        assert self.platforms is not None, "Add at least a platform before adding a target."
        for platform in self.platforms:
            self.target.add_platform_defs(platform.name, platform.definitions)
        self.cache_hints = [self.target.get_arch()]
        # self.resolve_chache_refs()

    def add_platform(self, platform):
        """Setter for the platform instance."""
        self.platforms = [platform]
        if self.backend:
            self.backend.add_platform_defs(platform.name, platform.definitions)
        if self.framework:
            self.framework.add_platform_defs(platform.name, platform.definitions)

    def add_platforms(self, platforms):
        """Setter for the list of platforms."""
        self.platforms = platforms
        for platform in platforms:
            if self.backend:
                self.backend.add_platform_defs(platform.name, platform.definitions)
            if self.framework:
                self.framework.add_platform_defs(platform.name, platform.definitions)

    def add_postprocess(self, postprocess, append=False):
        """Setter for a postprocess instance."""
        if append:
            self.postprocesses.append(postprocess)
        else:
            self.postprocesses = [postprocess]

    def add_postprocesses(self, postprocesses, append=False):
        """Setter for the list of postprocesses."""
        if append:
            self.postprocesses.extend(postprocesses)
        else:
            self.postprocesses = postprocesses

    def add_feature(self, feature):
        """Setter for a feature instance."""
        self.features = [feature]
        self.run_features = self.process_features(self.features)
        self.run_config = filter_config(self.run_config, "run", self.DEFAULTS, self.REQUIRED)

    def add_features(self, features, append=False):
        """Setter for the list of features."""
        self.features = features if not append else self.features + features
        self.run_features = self.process_features(self.features)
        self.run_config = filter_config(self.run_config, "run", self.DEFAULTS, self.REQUIRED)

    def pick_model_frontend(self, model_hints, backend=None):
        assert len(model_hints) > 0
        if backend is None:
            self.frontends = [
                frontend for frontend in self.frontends if model_hints[0].formats[0] in frontend.input_formats
            ]
            assert len(self.frontends) > 0
            return model_hints[0]
        for model_hint in model_hints:
            if backend.supports_model(model_hint):
                self.frontends = [
                    frontend for frontend in self.frontends if model_hint.formats[0] in frontend.input_formats
                ]
                return model_hint
        return None

    def add_model_by_name(self, model_name, context=None):
        """Helper function to initialize and configure a model by its name."""
        assert context is not None, "Please supply a context"
        assert len(self.frontends) > 0, "Add a frontend to the run before adding a model"
        model_hints = lookup_models([model_name], frontends=self.frontends, context=context)
        assert len(model_hints) > 0, f"Model with name '{model_name}' not found"
        model_hint = self.pick_model_frontend(model_hints, backend=self.backend)
        assert model_hint is not None, "Unable to pick a suitable model"
        self.add_model(model_hint)

    def add_frontend_by_name(self, frontend_name, context=None):
        """Helper function to initialize and configure a frontend by its name."""
        self.add_frontends_by_name([frontend_name], context=context)

    def add_frontends_by_name(self, frontend_names, context=None):
        """Helper function to initialize and configure frontends by their names."""
        frontends = []
        for name in frontend_names:
            assert context is not None and context.environment.has_frontend(
                name
            ), f"The frontend '{name}' is not enabled for this environment"
            frontends.append(self.init_component(SUPPORTED_FRONTENDS[name], context=context))
        self.add_frontends(frontends)

    def add_backend_by_name(self, backend_name, context=None):
        """Helper function to initialize and configure a backend by its name."""
        assert context is not None and context.environment.has_backend(
            backend_name
        ), f"The backend '{backend_name}' is not enabled for this environment"
        self.add_backend(self.init_component(SUPPORTED_BACKENDS[backend_name], context=context))
        framework_name = self.backend.framework  # TODO: does this work?
        assert context.environment.has_framework(
            framework_name
        ), f"The framework '{framework_name}' is not enabled for this environment"
        self.add_framework(self.init_component(SUPPORTED_FRAMEWORKS[framework_name], context=context))

    def add_target_by_name(self, target_name, context=None):
        """Helper function to initialize and configure a target by its name."""
        # We can not use the following code snipped are platform targets may be resolved dynamically
        # assert context is not None and context.environment.has_target(
        #     target_name
        # ), f"The target '{target_name}' is not enabled for this environment"
        assert len(self.platforms) > 0, "Please add a platform to the run before adding the target"
        self.add_target(self.init_component(self.target_platform.create_target(target_name), context=context))

    def add_platform_by_name(self, platform_name, context=None):
        """Helper function to initialize and configure a platform by its name."""
        self.add_platforms_by_name([platform_name], context=context)

    def add_platforms_by_name(self, platform_names, context=None):
        """Helper function to initialize and configure platforms by their names."""
        platforms = []
        for name in platform_names:
            assert context is not None and context.environment.has_platform(
                name
            ), f"The platform '{name}' is not enabled for this environment"
            platforms.append(self.init_component(get_platforms()[name], context=context))
        self.add_platforms(platforms)

    def add_postprocess_by_name(self, postprocess_name, append=True, context=None):
        """Helper function to initialize and configure a postprocesses by its name."""
        self.add_postprocesses_by_name([postprocess_name], append=append, context=context)

    def add_postprocesses_by_name(self, postprocess_names, append=False, context=None):
        """Helper function to initialize and configure postprocesses by their names."""
        postprocesses = []
        for name in postprocess_names:
            # TODO: ?
            # assert context is not None and context.environment.has_postprocess(
            #     postprocess_name
            # ), f"The postprocess '{postprocess_name}' is not enabled for this environment"
            postprocesses.append(self.init_component(SUPPORTED_POSTPROCESSES[name], context=context))
        self.add_postprocesses(postprocesses, append=append)

    def add_feature_by_name(self, feature_name, append=True, context=None):
        """Helper function to initialize and configure a feature by its name."""
        self.add_features_by_name([feature_name], context=context, append=append)

    def add_features_by_name(self, feature_names, append=False, context=None):
        """Helper function to initialize and configure features by their names."""
        features = []
        for feature_name in feature_names:
            available_features = get_available_features(feature_name=feature_name)
            for feature_cls in available_features:
                features.append(self.init_component(feature_cls, context=context))
        self.add_features(features, append=append)

    @property
    def export_optional(self):
        """Get export_optional property."""
        return bool(self.run_config["export_optional"])

    def __repr__(self):
        probs = []
        if self.model:
            probs.append(str(self.model))
        if len(self.platforms) > 0:
            probs.append(str(self.platforms[0] if len(self.platforms) == 1 else self.platforms))
        if len(self.frontends) > 0:
            probs.append(str(self.frontends[0] if len(self.frontends) == 1 else self.frontends))
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

    @property
    def frontend(self):
        assert len(self.frontends) > 0, "Not frontend is available for this run."
        return self.frontends[0]

    def toDict(self):
        """Utility not implemented yet. (TODO: remove?)"""
        raise NotImplementedError

    def export_stage(self, stage, optional=False, subdir=False):
        """Export stage artifacts of this run to its directory."""
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
                    # extract = artifact.fmt == ArtifactFormat.MLF and not isinstance(self.platform, MicroTvmPlatform)
                    artifact.export(self.dir)
                    # Keep the tar as well as the extracted files
                    if extract:
                        artifact.export(self.dir, extract=True)

    def postprocess(self):
        """Postprocess the 'run'."""
        logger.debug("%s Processing stage POSTPROCESS", self.prefix)
        self.lock()
        # assert self.completed[RunStage.RUN]  # Alternative: allow to trigger previous stages recursively as a fallback

        self.artifacts_per_stage[RunStage.POSTPROCESS] = []
        temp_report = self.get_report()
        for postprocess in self.postprocesses:
            if isinstance(postprocess, RunPostprocess):
                artifacts = postprocess.post_run(temp_report)
                if artifacts is not None:
                    self.artifacts_per_stage[RunStage.POSTPROCESS].extend(artifacts)
        self.report = temp_report

        self.completed[RunStage.POSTPROCESS] = True
        self.unlock()

    def run(self):
        """Run the 'run' using the defined target."""
        logger.debug("%s Processing stage RUN", self.prefix)
        self.lock()
        # Alternative: drop artifacts of higher stages when re-triggering a lower one?
        if self.has_stage(RunStage.COMPILE):
            assert self.completed[RunStage.COMPILE]
            self.export_stage(RunStage.COMPILE, optional=self.export_optional)
            elf_artifact = self.artifacts_per_stage[RunStage.COMPILE][0]
            self.target.generate_metrics(elf_artifact.path)
        else:
            assert self.completed[RunStage.BUILD]  # Used for tvm platform
            self.export_stage(RunStage.BUILD, optional=self.export_optional)
            shared_object_artifact = self.artifacts_per_stage[RunStage.BUILD][0]
            self.target.generate_metrics(shared_object_artifact.path, num=self.num)
        self.artifacts_per_stage[RunStage.RUN] = self.target.artifacts

        self.completed[RunStage.RUN] = True
        self.unlock()

    def compile(self):
        """Compile the target software for the run."""
        logger.debug("%s Processing stage COMPILE", self.prefix)
        self.lock()
        assert self.completed[RunStage.BUILD]

        self.export_stage(RunStage.BUILD, optional=self.export_optional)
        codegen_dir = self.dir
        data_file = None
        for artifact in self.artifacts_per_stage[RunStage.LOAD]:
            if artifact.name == "data.c":
                artifact.export(self.dir)
                data_file = Path(self.dir) / "data.c"
        self.compile_platform.generate_elf(codegen_dir, self.target, num=self.num, data_file=data_file)
        self.artifacts_per_stage[RunStage.COMPILE] = self.compile_platform.artifacts

        self.completed[RunStage.COMPILE] = True
        self.unlock()

    def build(self):
        """Process the run using the choosen backend."""
        logger.debug("%s Processing stage BUILD", self.prefix)
        self.lock()
        assert (not self.has_stage(RunStage.TUNE)) or self.completed[RunStage.TUNE]

        if self.target_to_backend:
            assert self.target is not None, "Config target_to_backend can only be used if a target was provided"
            cfg = self.target.get_backend_config(self.backend.name)  # Do not expect a backend prefix here
            if len(cfg) > 0:
                logger.debug("Updating backend config based on given target.")
                self.backend.config.update(cfg)

        self.export_stage(RunStage.LOAD, optional=self.export_optional)  # Not required anymore?
        model_artifact = self.artifacts_per_stage[RunStage.LOAD][0]
        if not model_artifact.exported:
            model_artifact.export(self.dir)
        self.backend.load_model(model=model_artifact.path)
        if self.has_stage(RunStage.TUNE):
            self.export_stage(RunStage.TUNE, optional=self.export_optional)
            if len(self.artifacts_per_stage[RunStage.TUNE]) > 0:
                assert self.backend.tuner is not None
                tuning_artifact = self.artifacts_per_stage[RunStage.TUNE][0]
                if not tuning_artifact.exported:
                    tuning_artifact.export(self.dir)
                self.backend.tuning_records = tuning_artifact.path

        # TODO: allow raw data as well as filepath in backends
        self.backend.generate_code()
        self.artifacts_per_stage[RunStage.BUILD] = self.backend.artifacts

        self.completed[RunStage.BUILD] = True
        self.unlock()

    def tune(self):
        """Tune the run using the choosen backend (if supported)."""
        logger.debug("%s Processing stage TUNE", self.prefix)
        self.lock()
        assert self.completed[RunStage.LOAD]

        self.export_stage(RunStage.LOAD, optional=self.export_optional)
        model_artifact = self.artifacts_per_stage[RunStage.LOAD][0]
        if not model_artifact.exported:
            model_artifact.export(self.dir)

        # TODO: allow raw data as well as filepath in backends
        self.backend.load_model(model=model_artifact.path)
        self.backend.tune_model()
        if self.backend.tuner is not None:
            res = self.backend.tuner.get_results()
            if res:
                self.artifacts_per_stage[RunStage.TUNE] = res
            else:
                self.artifacts_per_stage[RunStage.TUNE] = []
        else:
            self.artifacts_per_stage[RunStage.TUNE] = []

        self.completed[RunStage.TUNE] = True
        self.unlock()

    def load(self):
        """Load the model using the given frontend."""
        logger.debug("%s Processing stage LOAD", self.prefix)
        self.lock()
        # assert self.completed[RunStage.NOP]

        self.frontend.generate_models(self.model)
        # The following is very very dirty but required to update arena sizes via model metadata...
        cfg_new = {}
        data_artifact = self.frontend.process_metadata(self.model, cfg=cfg_new)
        if len(cfg_new) > 0:
            for key, value in cfg_new.items():
                component, name = key.split(".")[:2]
                if self.backend is not None and component == self.backend.name:
                    self.backend.config[name] = value
                else:
                    for platform in self.platforms:
                        if platform is not None and component == platform.name:
                            platform.config[name] = value
                self.config[key] = value
        self.artifacts_per_stage[RunStage.LOAD] = self.frontend.artifacts
        if data_artifact:
            self.artifacts_per_stage[RunStage.LOAD].append(data_artifact)

        self.completed[RunStage.LOAD] = True
        self.unlock()

    def process(self, until=RunStage.RUN, skip=None, export=False):
        """Process the run until a given stage."""
        skip = skip if skip is not None else []
        if until == RunStage.DONE:
            until = RunStage.DONE - 1
        start = self.next_stage  # self.stage hold the max finished stage
        if until < start:
            logger.debug("%s Nothing to do", self.prefix)
            return self.get_report()

        if start > RunStage.NOP:
            logger.debug(
                # self.prefix + "Processing run until stage %s: %s",
                "%s Continuing run from stage %s until stage %s",
                self.prefix,
                str(RunStage(start).name),
                str(RunStage(until).name),
                # str(self),
            )
        else:
            logger.debug(
                # self.prefix + "Processing run until stage %s: %s",
                "%s Processing run until stage %s",
                self.prefix,
                str(RunStage(until).name),
                # str(self),
            )
        for stage in range(start, until + 1):
            if not self.has_stage(stage):
                continue
            if stage in skip:
                logger.debug("%s Skipping stage %s", self.prefix, str(RunStage(stage).name))
                continue
            if stage < RunStage.POSTPROCESS:
                self.report = None  # Regenerate report if earlier stages are executed
            stage_funcs = {
                RunStage.NOP: lambda *args, **kwargs: None,  # stage already done
                RunStage.LOAD: self.load,
                RunStage.TUNE: self.tune,
                RunStage.BUILD: self.build,
                RunStage.COMPILE: self.compile,
                RunStage.RUN: self.run,
                RunStage.POSTPROCESS: self.postprocess,
            }
            func = stage_funcs[stage]
            if func:
                self.failing = False
                try:
                    func()
                except Exception as e:
                    self.failing = True
                    if self.locked:
                        self.unlock()
                    logger.exception(e)
                    run_stage = RunStage(stage).name
                    logger.error("%s Run failed at stage '%s', aborting...", self.prefix, run_stage)
                    break
            # self.stage = stage  # FIXME: The stage_func should update the stage intead?
        report = self.get_report()
        if export:
            self.export(optional=self.export_optional)  # TODO: set to flase?
            report_file = Path(self.dir) / "report.csv"
            report.export(report_file)
        return report

    def write_run_file(self):
        """Create a run.txt file which contains information used to reconstruct the run based
        on its properties at a later point in time."""
        logger.debug("%s Writing run file", self.prefix)
        filename = self.dir / "run.txt"
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(str(self))

    @property
    def prefix(self):
        """Get prefix property."""
        return (
            (f"[session-{self.session.idx}] [run-{self.idx}]" if self.session else f"[run-{self.idx}]")
            if self.idx is not None
            else ""
        )

    def get_all_feature_names(self):
        """Return list of feature names for this run."""
        return [feature.name for feature in self.features]

    def get_all_postprocess_names(self):
        """Return list of postprocess names for this run."""
        return [postprocess.name for postprocess in self.postprocesses]

    def get_frontend_name(self):
        """Return frontend name(s) for this run."""
        ret = [frontend.name for frontend in self.frontends]
        return ret[0] if len(ret) == 1 else ret

    def get_platform_name(self):
        """Return platform name(s) for this run."""
        used = list(set([self.compile_platform, self.target_platform]))  # TODO: build_platform, tune_platform
        ret = [platform.name for platform in used]
        return ret[0] if len(ret) == 1 else ret

    def get_all_configs(self, omit_paths=False, omit_defaults=False, omit_globals=False):
        """Return dict with component-specific and global configuration for this run."""

        def has_prefix(key):
            """Returns true if the configuration key does not have global scope."""
            return "." in key

        def config_helper(obj, prefix=None):
            """Helper to access the configuration of a given component object."""
            if prefix:
                name = prefix
            else:
                assert hasattr(obj, "name")
                name = obj.name
            omit_list = []
            defaults = obj.DEFAULTS
            for key, value in obj.config.items():
                if omit_defaults and key in defaults and defaults[key] == value:
                    omit_list.append(key)
            ret = {
                key if has_prefix(key) else f"{name}.{key}": value
                for key, value in obj.config.items()
                if key not in omit_list
            }
            if omit_paths:
                ret = {
                    key: value
                    for key, value in ret.items()
                    if not (isinstance(value, Path) or (isinstance(value, str) and Path(value).exists()))
                }
            return ret

        ret = {}
        if not omit_globals:
            ret.update(
                {key: value for key, value in self.config.items() if not has_prefix(key)}
            )  # Only config without a prefix!
        for frontend in self.frontends:
            ret.update(config_helper(frontend))
        if self.backend:
            ret.update(config_helper(self.backend))
        if self.framework:
            ret.update(config_helper(self.framework))
        if self.target:
            ret.update(config_helper(self.target))
        for platform in self.platforms:
            ret.update(config_helper(platform))
        for postprocess in self.postprocesses:
            ret.update(config_helper(postprocess))
        return ret

    def get_report(self):
        """Returns teh complete report of this run."""
        if self.completed[RunStage.POSTPROCESS]:
            if self.report is not None:
                return (
                    self.report
                )  # Use postprocessed report instead of generating a new one (TODO: find a better approach)
        # TODO: config or args for stuff like (session id) and run id as well as detailed features and configs
        report = Report()
        pre = {}
        if self.session is not None:
            pre["Session"] = self.session.idx
        if self.idx is not None:
            pre["Run"] = self.idx
        if self.model:
            pre["Model"] = self.model.name
        if len(self.frontends) > 0:
            pre["Frontend"] = self.get_frontend_name()
        if self.framework:
            pre["Framework"] = self.framework.name
        if self.backend:
            pre["Backend"] = self.backend.name
        if len(self.platforms) > 0:
            pre["Platform"] = self.get_platform_name()
        if self.target:
            pre["Target"] = self.target.name
        pre["Num"] = self.num
        post = {}
        post["Features"] = self.get_all_feature_names()
        post["Config"] = self.get_all_configs(omit_paths=True, omit_defaults=True, omit_globals=True)
        post["Postprocesses"] = self.get_all_postprocess_names()
        post["Comment"] = self.comment if len(self.comment) > 0 else "-"
        if self.failing:
            post["Failing"] = True

        self.export_stage(RunStage.RUN, optional=self.export_optional)

        metrics = Metrics()
        if RunStage.COMPILE in self.artifacts_per_stage:
            if len(self.artifacts_per_stage[RunStage.COMPILE]) > 1:  # TODO: look for artifact of type metrics instead
                compile_metrics_artifact = lookup_artifacts(
                    self.artifacts_per_stage[RunStage.COMPILE], name="metrics.csv"
                )[0]
                compile_metrics = Metrics.from_csv(compile_metrics_artifact.content)
                metrics = compile_metrics

        if RunStage.RUN in self.artifacts_per_stage:
            run_metrics_artifact = lookup_artifacts(self.artifacts_per_stage[RunStage.RUN], name="metrics.csv")[0]
            run_metrics = Metrics.from_csv(run_metrics_artifact.content)
            # Combine with compile metrics
            metrics_data = metrics.get_data()
            run_metrics_data = run_metrics.get_data()
            for key, value in metrics_data.items():
                if key not in run_metrics_data:
                    run_metrics.add(key, value)
            metrics = run_metrics

        main = metrics.get_data(include_optional=self.export_optional)
        report.set(pre=[pre], main=[main] if len(main) > 0 else {"Incomplete": [True]}, post=[post])
        return report

    def export(self, path=None, optional=False):
        """Write a run configuration to a disk."""
        logger.debug("%s Exporting run to disk", self.prefix)
        if path is not None:
            raise NotImplementedError
        assert not self.locked
        for stage in range(self.next_stage):
            if not self.has_stage(stage):
                continue
            self.export_stage(stage, optional=optional)

        self.write_run_file()


# TODO: implement close()? and use closing contextlib?
