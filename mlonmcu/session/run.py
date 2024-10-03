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
import itertools
import os
import copy
import tempfile
from pathlib import Path
from enum import IntEnum
from collections import defaultdict

from mlonmcu.logging import get_logger
from mlonmcu.artifact import ArtifactFormat, lookup_artifacts
from mlonmcu.config import str2bool
from mlonmcu.platform.platform import CompilePlatform, TargetPlatform, BuildPlatform, TunePlatform
from mlonmcu.report import Report  # TODO: move to mlonmcu.session.report
from mlonmcu.config import resolve_required_config, filter_config
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import get_matching_features, get_available_features
from mlonmcu.target.metrics import Metrics
from mlonmcu.models import SUPPORTED_FRONTENDS
from mlonmcu.models.model import Model, Program
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


def add_any(new, base=None, append=True):
    ret = []
    if append:
        if isinstance(base, list):
            ret = base

    if isinstance(new, list):
        ret.extend(new)
    else:
        ret.append(new)
    return ret


class Run:
    """A run is single model/backend/framework/target combination with a given set of features and configs."""

    FEATURES = {"autotune", "target_optimized", "validate_new"}

    DEFAULTS = {
        "export_optional": False,
        "tune_enabled": False,
        "target_to_backend": True,
        "target_optimized_layouts": False,
        "target_optimized_schedules": False,
        "stage_subdirs": False,
    }

    REQUIRED = set()
    OPTIONAL = set()

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
        self.archived = archived
        self.session = session
        self.postprocesses = postprocesses if postprocesses else []
        self.comment = comment
        # self.stage = RunStage.NOP  # max executed stage
        self.completed = {stage: stage == RunStage.NOP for stage in RunStage}

        self.directories = {}
        # self.init_directory()
        self.target = target
        self.cache_hints = []
        self.config = config if config else {}
        self.features = features if features else []
        self.run_config = {}
        self.run_features = self.process_features(features)
        self.run_config = filter_config(self.config, "run", self.DEFAULTS, self.OPTIONAL, self.REQUIRED)
        self.sub_names = []
        self.sub_parents = {}
        self.result = None
        self.failing = False  # -> RunStatus
        self.reason = None
        self.failed_stage = None
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
            feature.used = True
            tmp_run_config = {f"run.{key}": value for key, value in self.run_config.items()}
            feature.add_run_config(tmp_run_config)
            self.run_config = filter_config(tmp_run_config, "run", self.DEFAULTS, self.OPTIONAL, self.REQUIRED)
        return features

    @property
    def tune_enabled(self):
        """Get tune_enabled property."""
        value = self.run_config["tune_enabled"]
        return str2bool(value)

    @property
    def target_to_backend(self):
        """Get target_to_backend property."""
        value = self.run_config["target_to_backend"]
        return str2bool(value)

    @property
    def target_optimized_layouts(self):
        """Get target_optimized_layouts property."""
        value = self.run_config["target_optimized_layouts"]
        return str2bool(value)

    @property
    def target_optimized_schedules(self):
        """Get target_optimized_schedules property."""
        value = self.run_config["target_optimized_schedules"]
        return str2bool(value)

    @property
    def export_optional(self):
        """Get export_optional property."""
        value = self.run_config["export_optional"]
        return str2bool(value)

    @property
    def stage_subdirs(self):
        value = self.run_config["stage_subdirs"]
        return str2bool(value)

    @property
    def build_platform(self):
        """Get platform for build stage."""
        if self.backend is not None and (
            hasattr(self.backend, "platform") and isinstance(self.backend.platform, BuildPlatform)
        ):
            return self.backend.platform
        for platform in self.platforms:
            if isinstance(platform, BuildPlatform):
                return platform
        return None

    @property
    def tune_platform(self):
        """Get platform for tune stage."""
        if self.backend is not None and (
            hasattr(self.backend, "platform") and isinstance(self.backend.platform, TunePlatform)
        ):
            return self.backend.platform
        for platform in self.platforms:
            if isinstance(platform, TunePlatform):
                return platform
        return None

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

    @property
    def last_stage(self):
        """Determines the next not yet completed stage. Returns RunStage.DONE if already completed."""
        last = None
        for stage in RunStage:
            if self.has_stage(stage):
                if not self.completed[stage.value]:
                    return last
                last = stage
        return None

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
                # The stage_subdirs setting is ignored here because platforms can be multi-stage!
                # platform.init_directory(path=Path(self.dir) / platform.name)
                if platform in self.directories:
                    continue
                platform_dir = Path(self.dir) / platform.name
                if platform.init_directory(path=platform_dir):
                    self.directories[platform.name] = platform_dir
            # if target not in self.directories:
            #     target_dir = Path(self.dir) /target.name
            #     if target.init_directory(path=target_dir)
            #         self.directories[target.name] = target_dir

            # TODO: other components

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "session":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def copy(self):
        """Create a new run based on this instance."""
        new = copy.deepcopy(self)
        if self.session:
            new_idx = self.session.request_run_idx()
            new.idx = new_idx
            # self.init_directory()
        return new

    def init_component(self, component_cls, context=None):
        """Helper function to create and configure a MLonMCU component instance for this run."""
        required_keys = component_cls.REQUIRED
        optional_keys = component_cls.OPTIONAL
        self.config.update(
            resolve_required_config(
                required_keys,
                optional=optional_keys,
                features=self.features,
                config=self.config,
                cache=context.cache if context else None,
                default_flags=context.environment.flags if context else None,
                hints=self.cache_hints,
            )
        )
        component_config = self.config.copy()  # TODOL get rid of this
        return component_cls(features=self.features, config=component_config)

    def add_model(self, model):
        """Setter for the model instance."""
        self.model = model
        assert model is not None
        self.model.config = filter_config(self.config, self.model.name, self.model.DEFAULTS, set(), set())
        for platform in self.platforms:
            self.model.add_platform_config(platform, platform.config)
            self.model.add_platform_defs(platform, platform.definitions)
        # TODO: update after load stage
        # for platform in self.platforms:
        #     self.add_platform_defs(platform.name, platform.definitions)

    def add_frontend(self, frontend, append=True):
        """Setter for the frontend instance."""
        self.frontends = add_any(frontend, self.frontends, append=append)
        for frontend in self.frontends:
            for platform in self.platforms:
                frontend.add_platform_config(platform, platform.config)
                frontend.add_platform_defs(platform.name, platform.definitions)

    def add_frontends(self, frontends, append=False):
        """Setter for the list of frontends."""
        self.frontends = add_any(frontends, self.frontends, append=append)
        for frontend in self.frontends:
            for platform in self.platforms:
                frontend.add_platform_config(platform, platform.config)
                frontend.add_platform_defs(platform.name, platform.definitions)

    def add_backend(self, backend):
        """Setter for the backend instance."""
        self.backend = backend
        # assert len(self.platforms) > 0, "Add at least a platform before adding a backend."
        if self.model is not None:
            if not isinstance(self.model, Model):
                self.backend = None
                return
            if not self.model.skip_check:
                assert self.backend.supports_model(self.model), (
                    "The added backend does not support the chosen model. "
                    "Add the backend before adding a model to find a suitable frontend."
                )
        for platform in self.platforms:
            self.backend.add_platform_config(platform.name, platform.config)
            self.backend.add_platform_defs(platform.name, platform.definitions)

    def add_framework(self, framework):
        """Setter for the framework instance."""
        self.framework = framework
        # assert len(self.platforms) > 0, "Add at least a platform before adding a framework."
        for platform in self.platforms:
            self.framework.add_platform_config(platform.name, platform.config)
            self.framework.add_platform_defs(platform.name, platform.definitions)

    def add_target(self, target):
        """Setter for the target instance."""
        self.target = target
        assert self.platforms is not None, "Add at least a platform before adding a target."
        for platform in self.platforms:
            self.target.add_platform_config(platform.name, platform.config)
            self.target.add_platform_defs(platform.name, platform.definitions)
        self.cache_hints = [self.target.get_arch()]
        # self.resolve_chache_refs()

    def add_platform(self, platform, append=True):
        """Setter for the platform instance."""
        self.platforms = add_any(platform, self.platforms, append=append)
        for frontend in self.frontends:
            self.frontend.add_platform_config(platform.name, platform.config)
            self.frontend.add_platform_defs(platform.name, platform.definitions)
        if self.model:
            self.model.add_platform_config(platform.name, platform.config)
            self.model.add_platform_defs(platform.name, platform.definitions)
        if self.backend:
            self.backend.add_platform_config(platform.name, platform.config)
            self.backend.add_platform_defs(platform.name, platform.definitions)
        if self.framework:
            self.framework.add_platform_config(platform.name, platform.config)
            self.framework.add_platform_defs(platform.name, platform.definitions)

    def add_platforms(self, platforms, append=False):
        """Setter for the list of platforms."""
        self.platforms = add_any(platforms, self.platforms, append=append)
        # TODO: check for duplicates?
        for platform in platforms:
            for frontend in self.frontends:
                self.frontend.add_platform_config(platform.name, platform.config)
                self.frontend.add_platform_defs(platform.name, platform.definitions)
            if self.model:
                self.model.add_platform_config(platform.name, platform.config)
                self.model.add_platform_defs(platform.name, platform.definitions)
            if self.backend:
                self.backend.add_platform_config(platform.name, platform.config)
                self.backend.add_platform_defs(platform.name, platform.definitions)
            if self.framework:
                self.framework.add_platform_config(platform.name, platform.config)
                self.framework.add_platform_defs(platform.name, platform.definitions)

    def add_postprocess(self, postprocess, append=True):
        """Setter for a postprocess instance."""
        self.postprocesses = add_any(postprocess, self.postprocesses, append=append)

    def add_postprocesses(self, postprocesses, append=False):
        """Setter for the list of postprocesses."""
        self.postprocesses = add_any(postprocesses, self.postprocesses, append=append)

    def add_feature(self, feature, append=True):
        """Setter for a feature instance."""
        self.features = add_any(feature, self.features, append=append)
        self.run_features = self.process_features(self.features)

    def add_features(self, features, append=False):
        """Setter for the list of features."""
        self.features = add_any(features, self.features, append=append)
        self.run_features = self.process_features(self.features)

    def add_model_by_name(self, model_name, context=None):
        """Helper function to initialize and configure a model by its name."""
        assert context is not None, "Please supply a context"
        assert len(self.frontends) > 0, "Add a frontend to the run before adding a model"
        model = None
        reasons = {}
        for frontend in self.frontends:
            if model is not None:
                break
            try:
                model_hints = frontend.lookup_models([model_name], config=self.config, context=context)
                # model_hints = lookup_models([model_name], frontends=self.frontends, context=context)
                for model_hint in model_hints:
                    if (
                        self.backend is None
                        or isinstance(model_hint, Program)
                        or (self.backend is not None and self.backend.supports_model(model_hint))
                    ):
                        self.frontends = [frontend]
                        assert model_hint is not None, "Unable to pick a suitable model"
                        model = model_hint
                        break
            except Exception as e:
                reasons[frontend.name] = str(e)
        if model is None:
            if reasons:
                logger.error("Lookup of model '%s' was not successfull. Reasons: %s", model_name, reasons)
            else:
                raise RuntimeError(f"Model with name '{model_name}' not found.")
        self.add_model(model)

    def add_frontend_by_name(self, frontend_name, context=None):
        """Helper function to initialize and configure a frontend by its name."""
        self.add_frontends_by_name([frontend_name], context=context)

    def add_frontends_by_name(self, frontend_names, context=None):
        """Helper function to initialize and configure frontends by their names."""
        frontends = []
        reasons = {}
        for name in frontend_names:
            try:
                assert context is not None and context.environment.has_frontend(
                    name
                ), f"The frontend '{name}' is not enabled for this environment"
                frontends.append(self.init_component(SUPPORTED_FRONTENDS[name], context=context))
            except Exception as e:
                reasons[name] = str(e)
                continue
        assert len(frontends) > 0, "No compatible frontend was found"
        if len(frontends) == 0:
            if reasons:
                logger.error("Initialization of frontends was no successfull. Reasons: %s", reasons)
            else:
                raise RuntimeError("No compatible frontend was found.")
        self.add_frontends(frontends)

    def add_backend_by_name(self, backend_name, context=None):
        """Helper function to initialize and configure a backend by its name."""
        assert context is not None and context.environment.has_backend(
            backend_name
        ), f"The backend '{backend_name}' is not enabled for this environment"
        if self.build_platform:
            self.add_backend(self.init_component(self.build_platform.create_backend(backend_name), context=context))
        else:
            self.add_backend(self.init_component(SUPPORTED_BACKENDS[backend_name], context=context))
        if self.backend is None:
            return
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
            available_features = get_available_features(feature_name=feature_name, deps=True)
            # check for already added features
            temp = self.features + features if append else features
            added_names = [f.name for f in temp]
            for feature_name_, feature_cls_ in available_features.items():
                if feature_name_ in added_names:
                    continue
                features.append(self.init_component(feature_cls_, context=context))
        self.add_features(features, append=append)

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
        if self.features and len(self.features) > 0:
            probs.append(str(self.features))
        if self.config and len(self.config) > 0:
            probs.append(str(self.config))
        return "Run(" + ",".join(probs) + ")"

    @property
    def frontend(self):
        assert len(self.frontends) > 0, "Not frontend is available for this run."
        return self.frontends[0]

    @property
    def artifacts(self):
        sub = "default"
        ret = sum(list(itertools.chain([subs[sub] for stage, subs in self.artifacts_per_stage.items()])), [])
        return ret

    def get_all_sub_artifacts(self, sub, stage=None):
        if sub is None:
            return []
        assert sub in self.sub_names
        if stage is None:
            stage = self.last_stage
        parents = self.sub_parents[(stage, sub)]
        stage_, sub_ = parents
        artifacts = self.get_all_sub_artifacts(sub_, stage_) + self.artifacts_per_stage[stage][sub]
        return artifacts

    def toDict(self):
        """Utility not implemented yet. (TODO: remove?)"""
        raise NotImplementedError

    def export_stage(self, stage, optional=False):
        """Export stage artifacts of this run to its directory."""
        # TODO: per stage subdirs?
        subdir = self.stage_subdirs
        if stage in self.artifacts_per_stage:
            for name in self.artifacts_per_stage[stage]:
                artifacts = self.artifacts_per_stage[stage][name]
                for artifact in artifacts:
                    if not artifact.optional or optional:
                        dest = self.dir
                        if subdir:
                            stage_idx = int(stage)
                            dest = dest / "stages" / str(stage_idx)
                            # TODO: stages.txt for mapping between stage idx and name
                        if name not in ["", "default"]:
                            dest = dest / "sub" / name
                        dest.mkdir(parents=True, exist_ok=True)
                        extract = artifact.fmt in [ArtifactFormat.MLF, ArtifactFormat.ARCHIVE]
                        # extract = artifact.fmt == ArtifactFormat.MLF
                        # and not isinstance(self.platform, MicroTvmPlatform)
                        artifact.export(dest)
                        # Keep the tar as well as the extracted files
                        if extract:
                            artifact.export(dest, extract=True)

    def postprocess(self):
        """Postprocess the 'run'."""
        logger.debug("%s Processing stage POSTPROCESS", self.prefix)
        self.lock()
        # assert self.completed[RunStage.RUN]  # Alternative: allow to trigger previous stages recursively as a fallback

        def _merge_dicts_of_lists(*args):
            if len(args) == 0:
                return {}
            a = args[0]
            de = defaultdict(list, a)
            for b in args[1:]:
                for (
                    i,
                    j,
                ) in b.items():
                    de[i].extend(j)
            return dict(de)

        self.artifacts_per_stage[RunStage.POSTPROCESS] = {"default": []}
        temp_report = self.get_report()
        last_stage = self.last_stage
        assert last_stage is not None
        self.export_stage(last_stage, optional=self.export_optional)
        for name in self.artifacts_per_stage[last_stage]:
            merged = {"default": []}
            before = self.get_all_sub_artifacts(name)
            for postprocess in self.postprocesses:
                if isinstance(postprocess, RunPostprocess):
                    artifacts = postprocess.post_run(temp_report, before)
                    before.extend(artifacts)
                    if artifacts is None:
                        artifacts = []
                    new = {}
                    if isinstance(artifacts, dict):
                        new.update(
                            {
                                (
                                    key
                                    if name in ["", "default"]
                                    else (f"{name}_{key}" if key not in ["", "default"] else name)
                                ): value
                                for key, value in artifacts.items()
                            }
                        )
                    else:
                        new.update({name if name in ["", "default"] else f"{name}": artifacts})
                    merged = _merge_dicts_of_lists(merged, new)
            self.artifacts_per_stage[RunStage.POSTPROCESS].update(merged)
            self.sub_parents.update({(RunStage.POSTPROCESS, key): (self.last_stage, name) for key in merged.keys()})
        self.sub_names.extend(self.artifacts_per_stage[RunStage.POSTPROCESS])
        self.sub_names = list(set(self.sub_names))

        self.report = temp_report

        self.completed[RunStage.POSTPROCESS] = True
        self.unlock()
        self.export_stage(RunStage.POSTPROCESS, optional=self.export_optional)

    def run(self):
        """Run the 'run' using the defined target."""
        logger.debug("%s Processing stage RUN", self.prefix)
        self.lock()
        # Alternative: drop artifacts of higher stages when re-triggering a lower one?
        self.artifacts_per_stage[RunStage.RUN] = {}
        if self.has_stage(RunStage.COMPILE):
            assert self.completed[RunStage.COMPILE]
            self.export_stage(RunStage.COMPILE, optional=self.export_optional)
            for name in self.artifacts_per_stage[RunStage.COMPILE]:
                elf_artifact = self.artifacts_per_stage[RunStage.COMPILE][name][0]
                artifacts = self.target.generate_artifacts(elf_artifact.path)
                if isinstance(artifacts, dict):
                    new = {
                        (
                            key
                            if name in ["", "default"]
                            else (f"{name}_{key}" if key not in ["", "default"] else name)
                        ): value
                        for key, value in artifacts.items()
                    }
                else:
                    new = {name if name in ["", "default"] else f"{name}": artifacts}
                self.artifacts_per_stage[RunStage.RUN].update(new)
                self.sub_parents.update({(RunStage.RUN, key): (self.last_stage, name) for key in new.keys()})
        else:
            assert self.completed[RunStage.BUILD]  # Used for tvm platform
            self.export_stage(RunStage.BUILD, optional=self.export_optional)
            for name in self.artifacts_per_stage[RunStage.BUILD]:
                shared_object_artifact = self.artifacts_per_stage[RunStage.BUILD][name][0]
                artifacts = self.target.generate_artifacts(shared_object_artifact.path)
                if isinstance(artifacts, dict):
                    new = {
                        (
                            key
                            if name in ["", "default"]
                            else (f"{name}_{key}" if key not in ["", "default"] else name)
                        ): value
                        for key, value in artifacts.items()
                    }
                else:
                    new = {name if name in ["", "default"] else f"{name}": artifacts}
                self.artifacts_per_stage[RunStage.RUN].update(new)
                self.sub_parents.update({(RunStage.RUN, key): (self.last_stage, name) for key in new.keys()})
        self.sub_names.extend(self.artifacts_per_stage[RunStage.RUN])
        self.sub_names = list(set(self.sub_names))

        self.completed[RunStage.RUN] = True
        self.unlock()

    def compile(self):
        """Compile the target software for the run."""
        logger.debug("%s Processing stage COMPILE", self.prefix)
        self.lock()
        if isinstance(self.model, Model):
            assert self.completed[RunStage.BUILD]

            self.export_stage(RunStage.BUILD, optional=self.export_optional)
            self.artifacts_per_stage[RunStage.COMPILE] = {}
            for name in self.artifacts_per_stage[RunStage.BUILD]:
                codegen_dir = self.dir if not self.stage_subdirs else (self.dir / "stages" / str(int(RunStage.BUILD)))
                if name not in ["", "default"]:
                    codegen_dir = codegen_dir / "sub" / name
                # TODO!
                artifacts = self.compile_platform.generate_artifacts(
                    codegen_dir, self.target
                )  # TODO: has to go into different dirs
                # artifacts = self.compile_platform.artifacts
                if isinstance(artifacts, dict):
                    new = {
                        (
                            key
                            if name in ["", "default"]
                            else (f"{name}_{key}" if key not in ["", "default"] else name)
                        ): value
                        for key, value in artifacts.items()
                    }
                else:
                    new = {name if name in ["", "default"] else f"{name}": artifacts}
                self.artifacts_per_stage[RunStage.COMPILE].update(new)
                self.sub_parents.update({(RunStage.COMPILE, key): (self.last_stage, name) for key in new.keys()})
        else:
            assert self.completed[RunStage.LOAD]
            self.artifacts_per_stage[RunStage.COMPILE] = {}
            codegen_dir = self.dir if not self.stage_subdirs else (self.dir / "stages" / str(int(RunStage.BUILD)))
            name = "default"
            artifacts = self.compile_platform.generate_artifacts(codegen_dir, self.target)
            if isinstance(artifacts, dict):
                new = {
                    key if name in ["", "default"] else (f"{name}_{key}" if key not in ["", "default"] else name): value
                    for key, value in artifacts.items()
                }
            else:
                new = {name if name in ["", "default"] else f"{name}": artifacts}
            self.artifacts_per_stage[RunStage.COMPILE].update(new)
            self.sub_parents.update({(RunStage.COMPILE, key): (self.last_stage, name) for key in new.keys()})
        self.sub_names.extend(self.artifacts_per_stage[RunStage.COMPILE])
        self.sub_names = list(set(self.sub_names))

        self.completed[RunStage.COMPILE] = True
        self.unlock()

    def build(self):
        """Process the run using the choosen backend."""
        logger.debug("%s Processing stage BUILD", self.prefix)
        self.lock()
        assert (not self.has_stage(RunStage.TUNE)) or self.completed[RunStage.TUNE]

        target_to_backend = self.target_to_backend and (self.target is not None)
        if self.backend.needs_target or self.target_optimized_layouts or self.target_optimized_schedules:
            assert self.target is not None, "Config target_to_backend can only be used if a target was provided"
            target_to_backend = True

        if target_to_backend:
            self.target.add_backend_config(
                self.backend.name,
                self.backend.config,
                optimized_layouts=self.target_optimized_layouts,
                optimized_schedules=self.target_optimized_schedules,
            )

        def _build():
            # TODO: allow raw data as well as filepath in backends
            artifacts = self.backend.generate_artifacts()
            if isinstance(artifacts, dict):
                new = {
                    key if name in ["", "default"] else (f"{name}_{key}" if key not in ["", "default"] else name): value
                    for key, value in artifacts.items()
                }
            else:
                new = {name if name in ["", "default"] else f"{name}": artifacts}
            self.artifacts_per_stage[RunStage.BUILD].update(new)
            self.sub_parents.update({(RunStage.BUILD, key): (self.last_stage, name) for key in new.keys()})

        self.artifacts_per_stage[RunStage.BUILD] = {}
        if self.has_stage(RunStage.TUNE):
            self.export_stage(RunStage.TUNE, optional=self.export_optional)
            for name in self.artifacts_per_stage[RunStage.TUNE]:
                tune_stage_artifacts = self.artifacts_per_stage[RunStage.TUNE][name]
                tuning_artifact = lookup_artifacts(
                    tune_stage_artifacts, fmt=ArtifactFormat.TEXT, flags=["records"], first_only=True
                )
                if len(tuning_artifact) == 0:
                    # fallback for metascheduler
                    tuning_artifact = lookup_artifacts(
                        tune_stage_artifacts,
                        fmt=ArtifactFormat.ARCHIVE,
                        flags=["records", "metascheduler"],
                        first_only=True,
                    )
                    if len(tuning_artifact) == 0:
                        continue
                tuning_artifact = tuning_artifact[0]
                if not tuning_artifact.exported:
                    tuning_artifact.export(self.dir)
                tuner_name = tuning_artifact.flags[-1]  # TODO: improve
                self.backend.set_tuning_records(tuning_artifact.path, tuner_name=tuner_name)
                candidate = (RunStage.TUNE, name)
                assert candidate in self.sub_parents
                parent_stage, parent_name = self.sub_parents[candidate]
                assert parent_stage == RunStage.LOAD
                assert parent_name in self.artifacts_per_stage[RunStage.LOAD]
                load_stage_artifacts = self.artifacts_per_stage[parent_stage][parent_name]
                model_artifact = lookup_artifacts(load_stage_artifacts, flags=["model"], first_only=True)
                assert len(model_artifact) > 0
                model_artifact = model_artifact[0]
                if not model_artifact.exported:
                    model_artifact.export(self.dir)
                input_shapes = None
                output_shapes = None
                input_types = None
                output_types = None
                if model_artifact.name.split(".", 1)[0] == self.model.name:
                    input_shapes = self.model.input_shapes
                    output_shapes = self.model.output_shapes
                    input_types = self.model.input_types
                    output_types = self.model.output_types
                self.backend.load_model(
                    model=model_artifact.path,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    input_types=input_types,
                    output_types=output_types,
                )
                _build()

        else:
            self.export_stage(RunStage.LOAD, optional=self.export_optional)  # Not required anymore?
            for name in self.artifacts_per_stage[RunStage.LOAD]:
                load_stage_artifacts = self.artifacts_per_stage[RunStage.LOAD][name]
                model_artifact = lookup_artifacts(load_stage_artifacts, flags=["model"], first_only=True)
                if len(model_artifact) == 0:
                    # TODO: This breaks because number of subs can not decrease...
                    continue
                assert len(model_artifact) == 1
                model_artifact = model_artifact[0]
                if not model_artifact.exported:
                    model_artifact.export(self.dir)
                input_shapes = None
                output_shapes = None
                input_types = None
                output_types = None
                if model_artifact.name.split(".", 1)[0] == self.model.name:
                    input_shapes = self.model.input_shapes
                    output_shapes = self.model.output_shapes
                    input_types = self.model.input_types
                    output_types = self.model.output_types
                self.backend.load_model(
                    model=model_artifact.path,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    input_types=input_types,
                    output_types=output_types,
                )
                _build()

        self.sub_names.extend(self.artifacts_per_stage[RunStage.BUILD])
        self.sub_names = list(set(self.sub_names))

        self.completed[RunStage.BUILD] = True
        self.unlock()

    def tune(self):
        """Tune the run using the choosen backend (if supported)."""
        logger.debug("%s Processing stage TUNE", self.prefix)
        self.lock()
        assert self.completed[RunStage.LOAD]

        target_to_backend = self.target_to_backend and (self.target is not None)
        if self.backend.needs_target or self.target_optimized_layouts or self.target_optimized_schedules:
            assert self.target is not None, "Backend needs target"
            target_to_backend = True

        if target_to_backend:
            self.target.add_backend_config(
                self.backend.name,
                self.backend.config,
                optimized_layouts=self.target_optimized_layouts,
                optimized_schedules=self.target_optimized_schedules,
            )

        self.export_stage(RunStage.LOAD, optional=self.export_optional)
        self.artifacts_per_stage[RunStage.TUNE] = {}
        for name in self.artifacts_per_stage[RunStage.LOAD]:
            model_artifact = self.artifacts_per_stage[RunStage.LOAD][name][0]
            # if not model_artifact.exported:
            #     model_artifact.export(self.dir)

            # TODO: allow raw data as well as filepath in backends
            assert self.tune_platform, "Autotuning requires a TunePlatform"
            res = self.tune_platform.tune_model(model_artifact.path, self.backend, self.target)
            new = {f"{name}": []}
            if res:
                if isinstance(res, dict):
                    new = {key if name in ["", "default"] else f"{name}_{key}": value for key, value in res.items()}
                else:
                    new = {name if name in ["", "default"] else f"{name}": res}
            self.artifacts_per_stage[RunStage.TUNE].update(new)
            self.sub_parents.update({(RunStage.TUNE, key): (RunStage.LOAD, name) for key in new.keys()})
        self.sub_names.extend(self.artifacts_per_stage[RunStage.TUNE])
        self.sub_names = list(set(self.sub_names))

        self.completed[RunStage.TUNE] = True
        self.unlock()

    def load(self):
        """Load the model using the given frontend."""
        logger.debug("%s Processing stage LOAD", self.prefix)
        self.lock()
        # assert self.completed[RunStage.NOP]

        artifacts = self.frontend.generate_artifacts(self.model)
        # The following is very very dirty but required to update arena sizes via model metadata...
        cfg_new = {}
        if isinstance(self.model, Model):
            artifacts_ = self.frontend.process_metadata(self.model, cfg=cfg_new)
            if artifacts_ is not None:
                if isinstance(artifacts, dict):
                    assert "default" in artifacts.keys()
                    artifacts["default"].extend(artifacts_)
                    # ignore subs for now
                else:
                    assert isinstance(artifacts, list)
                    artifacts.extend(artifacts_)
            if len(cfg_new) > 0:
                for key, value in cfg_new.items():
                    component, name = key.split(".")[:2]
                    if self.backend is not None and component == self.backend.name:
                        self.backend.config[name] = value
                    elif component == self.model.name:
                        # Do not overwrite user-provided shapes and types
                        if self.model.config[name] is None:
                            # self.model.config[name] = value
                            self.model.config = filter_config(
                                {key: value}, self.model.name, self.model.config, set(), set()
                            )
                    else:
                        for platform in self.platforms:
                            if platform is not None and component == platform.name:
                                platform.config[name] = value
                    self.config[key] = value
        if isinstance(artifacts, dict):
            self.artifacts_per_stage[RunStage.LOAD] = artifacts
        else:
            self.artifacts_per_stage[RunStage.LOAD] = {"default": artifacts}
        self.sub_names.extend(self.artifacts_per_stage[RunStage.LOAD])
        self.sub_names = list(set(self.sub_names))
        self.sub_parents.update(
            {(RunStage.LOAD, key): (None, None) for key in self.artifacts_per_stage[RunStage.LOAD].keys()}
        )

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
                    self.reason = e
                    if self.locked:
                        self.unlock()
                    logger.exception(e)
                    run_stage = RunStage(stage).name
                    self.failed_stage = run_stage
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

    def get_all_feature_names(self, only_used=True):
        """Return list of feature names for this run."""
        return [feature.name for feature in self.features if feature.used or not only_used]

    def get_all_postprocess_names(self):
        """Return list of postprocess names for this run."""
        return [postprocess.name for postprocess in self.postprocesses]

    def get_frontend_name(self):
        """Return frontend name(s) for this run."""
        ret = [frontend.name for frontend in self.frontends]
        return ret[0] if len(ret) == 1 else ret

    def get_platform_name(self):
        """Return platform name(s) for this run."""
        used = list(set([self.tune_platform, self.build_platform, self.compile_platform, self.target_platform]))
        ret = [platform.name for platform in used if platform is not None]
        return ret[0] if len(ret) == 1 else ret

    def get_reason_text(self):
        ret = str(type(self.reason).__name__) if self.reason else None
        if self.failed_stage:
            ret += " @ " + str(self.failed_stage)
        return ret

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
                    if not (
                        isinstance(value, Path)
                        or (isinstance(value, str) and len(str(value)) < 200 and Path(value).exists())
                    )
                }
            return ret

        ret = {}
        if not omit_globals:
            ret.update(
                {key: value for key, value in self.config.items() if not has_prefix(key)}
            )  # Only config without a prefix!
        ret.update(config_helper(self.model))
        for frontend in self.frontends:
            ret.update(config_helper(frontend))
        if self.backend:
            ret.update(config_helper(self.backend))
        if self.framework:
            ret.update(config_helper(self.framework))
        if self.target:
            self.target.reconfigure()
            ret.update(config_helper(self.target))
        for platform in self.platforms:
            ret.update(config_helper(platform))
        for feature in self.features:
            ret.update(config_helper(feature))
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
        post = {}
        post["Features"] = self.get_all_feature_names()
        # post["Config"] = self.get_all_configs(omit_paths=True, omit_defaults=True, omit_globals=True)
        post["Config"] = self.get_all_configs(omit_paths=True, omit_defaults=False, omit_globals=True)
        post["Postprocesses"] = self.get_all_postprocess_names()
        post["Comment"] = self.comment if len(self.comment) > 0 else "-"
        if self.failing:
            post["Failing"] = True
            reason_text = self.get_reason_text()
            if reason_text:
                post["Reason"] = reason_text

        self.export_stage(RunStage.RUN, optional=self.export_optional)

        subs = []
        # metrics = Metrics()
        metrics_by_sub = {}

        def metrics_helper(stage, subs):
            # if self.failing:
            #     return subs
            if stage in self.artifacts_per_stage:
                prev_metrics_by_sub = metrics_by_sub.copy()
                names = self.artifacts_per_stage[stage].keys()
                subs = names
                for name in self.artifacts_per_stage[stage]:
                    # metrics_by_sub[name] = Metrics()
                    if len(self.artifacts_per_stage[stage][name]) > 0:
                        filename = f"{stage.name.lower()}_metrics.csv"
                        metrics_artifact = lookup_artifacts(self.artifacts_per_stage[stage][name], name=filename)
                        if len(metrics_artifact) == 0:
                            continue
                        assert len(metrics_artifact) == 1
                        metrics_artifact = metrics_artifact[0]
                        metrics = Metrics.from_csv(metrics_artifact.content)
                        metrics_data = metrics.get_data(include_optional=self.export_optional)
                        # Combine with existing metrics
                        parents = self.sub_parents[(stage, name)]
                        parent_stage, parent_name = parents
                        if parent_name in prev_metrics_by_sub:
                            parent_metrics_data = prev_metrics_by_sub[parent_name].get_data(
                                include_optional=self.export_optional
                            )
                        else:
                            parent_metrics_data = {}
                        for key, value in parent_metrics_data.items():
                            if key not in metrics_data:
                                metrics.add(key, value)
                        metrics_by_sub[name] = metrics
            return subs

        for stage in range(RunStage.LOAD, RunStage.POSTPROCESS):
            subs_ = metrics_helper(RunStage(stage), subs)
            if len(subs_) < len(subs):
                # assert self.failing
                pass
            else:
                subs = subs_

        pres = []
        mains = []
        posts = []
        for sub in subs:
            if sub not in ["", "default"]:
                pres.append({**pre, "Sub": sub})
            else:
                pres.append(pre)
            if sub in metrics_by_sub:
                main = metrics_by_sub[sub].get_data(include_optional=self.export_optional)
            else:
                main = {}
            mains.append(main if len(main) > 0 else {"Incomplete": True})
            posts.append(post)  # TODO: omit for subs?
        report.set(pre=pres, main=mains, post=posts)
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
