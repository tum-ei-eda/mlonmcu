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

from contextlib import contextmanager
import itertools
import time
import copy
import shutil
import tempfile
from typing import Union, Optional
from pathlib import Path
from enum import IntEnum

import yaml

from mlonmcu.logging import get_logger
from mlonmcu.artifact import Artifact, ArtifactFormat, lookup_artifacts
from mlonmcu.config import Configurable, cfg, str2bool
from mlonmcu.platform.platform import CompilePlatform, TargetPlatform, BuildPlatform, TunePlatform
from mlonmcu.report import Report  # TODO: move to mlonmcu.session.report
from mlonmcu.config import resolve_required_config, filter_config
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import get_matching_features, get_available_features
from mlonmcu.target.metrics import Metrics
from mlonmcu.models import get_frontends
from mlonmcu.models.model import Model, Program
from mlonmcu.platform import get_platforms
from mlonmcu.flow import get_frameworks, get_backends

from .stage import Pipeline, StageId, StageContext, StageResult, STAGES, stage_key
from .postprocess import get_postprocesses

logger = get_logger()


class ModelLookupError(RuntimeError):
    """Raised when none of the enabled frontends can load a requested model."""


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


class RunInitializer:

    @staticmethod
    def from_file(src: Union[str, Path]):
        if not isinstance(src, Path):
            assert isinstance(src, str)
            src = Path(src)
        fmt = src.suffix
        assert len(fmt) > 0
        fmt = fmt[1:].lower()
        if fmt in ["yml", "yaml"]:
            with open(src, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        assert "runs" in data
        runs = data["runs"]
        if len(runs) == 0:
            raise RuntimeError("Empty run initializer")
        initializers = []
        for run in runs:
            initializer = RunInitializer(**run)
            initializer.frozen = True
            initializers.append(initializer)
        return initializers

    def __init__(
        self,
        idx=None,
        model_name=None,
        framework_name=None,
        frontend_names=None,
        backend_name=None,
        target_name=None,
        platform_names=None,
        feature_names=None,
        config=None,
        postprocess_names=None,
        comment=None,
        pipeline=None,
        # from_stage=None,
        # frozen=False,
    ):
        self.idx = idx
        self.model_name = model_name
        self.frontend_names = frontend_names
        self.framework_name = framework_name
        self.backend_name = backend_name
        self.platform_names = platform_names
        self.postprocess_names = postprocess_names
        self.comment = comment
        self.target_name = target_name
        self.config = config
        self.feature_names = feature_names
        self.pipeline = Pipeline(pipeline)
        self.frozen = False

    def _serialize(self):
        return {
            "runs": [
                {
                    "pipeline": self.pipeline.serialize(),
                    "idx": self.idx,
                    "model_name": self.model_name,
                    "frontend_names": self.frontend_names,
                    "framework_name": self.framework_name,
                    "backend_name": self.backend_name,
                    "platform_names": self.platform_names,
                    "postprocess_names": self.postprocess_names,
                    "comment": self.comment,
                    "target_name": self.target_name,
                    "config": self.config,
                    "feature_names": self.feature_names,
                    # not: frozen
                }
            ]
        }

    def save(self, dest: Union[str, Path], fmt: Optional[str] = None):
        if not isinstance(dest, Path):
            assert isinstance(dest, str)
            dest = Path(dest)
        data = self._serialize()
        if fmt is None:
            fmt = dest.suffix
            assert fmt
            if fmt[0] == ".":
                fmt = fmt[1:]
        if fmt.lower() in ["yml", "yaml"]:
            with open(dest, "w") as f:
                yaml.dump(data, f, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def realize(self, context=None):
        assert context is not None
        # Spawned process-pool workers receive a minimal context rather than
        # opening a full context (which normally loads extensions).
        from mlonmcu.context.context import MlonMcuContext, MlonMcuContextMinimal

        if isinstance(context, MlonMcuContextMinimal):
            # Also load overrides of known names in spawned workers. Forked
            # workers retain their inherited registrations without reloading.
            MlonMcuContext.load_extensions(context, skip_loaded=True)
        run = Run(
            idx=self.idx,
            config=self.config,
            pipeline=self.pipeline.serialize(),
        )
        if self.comment:
            run.comment = self.comment
        if self.feature_names:
            run.add_features_by_name(self.feature_names, context=context)
        if self.frontend_names:
            run.add_frontends_by_name(self.frontend_names, context=context)
        if self.model_name:
            run.add_model_by_name(self.model_name, context=context)
        if self.platform_names:
            run.add_platforms_by_name(self.platform_names, context=context)
        if self.backend_name:
            run.add_backend_by_name(self.backend_name, context=context)
        if self.target_name:
            run.add_target_by_name(self.target_name, context=context)
        if self.postprocess_names:
            run.add_postprocesses_by_name(self.postprocess_names, context=context)
        return run

    def has_target(self):
        return self.target_name is not None

    def add_model_by_name(self, model_name, context=None):
        assert self.model_name is None
        self.model_name = model_name

    def add_frontend_by_name(self, frontend_name, context=None):
        self.add_frontends_by_name([frontend_name])

    def add_frontends_by_name(self, frontend_names, context=None):
        if self.frontend_names is None:
            self.frontend_names = []
        self.frontend_names.extend(frontend_names)

    def add_backend_by_name(self, backend_name, context=None):
        assert self.backend_name is None
        self.backend_name = backend_name

    def add_target_by_name(self, target_name, context=None):
        # assert self.target_name is None
        self.target_name = target_name

    def add_platform_by_name(self, platform_name, context=None):
        self.add_platforms_by_name([platform_name])

    def add_platforms_by_name(self, platform_names, context=None):
        if self.platform_names is None:
            self.platform_names = []
        self.platform_names.extend(platform_names)

    def add_postprocess_by_name(self, postprocess_name, context=None):
        self.add_postprocesses_by_name([postprocess_name])

    def add_postprocesses_by_name(self, postprocess_names, context=None):
        if self.postprocess_names is None:
            self.postprocess_names = []
        self.postprocess_names.extend(postprocess_names)

    def add_feature_by_name(self, feature_name, context=None):
        self.add_features_by_name([feature_name])

    def add_features_by_name(self, feature_names, context=None):
        if self.feature_names is None:
            self.feature_names = []
        self.feature_names.extend(feature_names)

    def copy(self, session=None):
        """Create a new runinitializer based on this instance."""
        new = copy.deepcopy(self)
        assert session is not None, "Run.copy() needs session"
        if session:
            new_idx = session.request_run_idx()
            new.idx = new_idx
            # self.init_directory()
        return new


class RunResult:
    # def __init__(self, run: "Run", session: "Session"):
    def __init__(self, run: "Run"):
        self.idx = run.idx
        self.dir = run.dir
        self.failing = run.failing
        self.failed_stage = run.failed_stage
        self.reason = run.reason
        # self.report = run.get_report(session=session)
        self.report = run.get_report()
        # self.artifacts_per_stage = {}
        # self.stage = RunStage.NOP  # max executed stage
        # self.completed = {stage: stage == RunStage.NOP for stage in RunStage}
        # self.directories = {}

    def get_report(self, session=None):
        # TODO: read only?!
        # if session is not None:
        #     pre = self.report.pre_df
        #     pre["Session"] = session.idx
        #     self.report.pre_df = pre
        return self.report


class Run(Configurable):
    """A run is single model/backend/framework/target combination with a given set of features and configs."""

    FEATURES = {"autotune", "target_optimized", "validate_new"}

    export_optional = cfg(False, cast=str2bool, source="run_config")
    tune_enabled = cfg(False, cast=str2bool, source="run_config")
    target_to_backend = cfg(True, cast=str2bool, source="run_config")
    target_optimized_layouts = cfg(False, cast=str2bool, source="run_config")
    target_optimized_schedules = cfg(False, cast=str2bool, source="run_config")
    stage_subdirs = cfg(False, cast=str2bool, source="run_config")
    profile_stages = cfg(False, cast=str2bool, source="run_config")

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
        comment="",
        pipeline=None,
    ):
        self.idx = idx
        self.model = model  # Model name, not object?
        self.frontends = frontends if frontends is not None else []
        self.framework = framework  # ???
        self.backend = backend
        self.platforms = platforms if platforms is not None else []
        self.artifacts_per_stage = {}
        self.archived = archived
        self.postprocesses = postprocesses if postprocesses else []
        self.comment = comment
        # self.stage = RunStage.NOP  # max executed stage
        self.pipeline = Pipeline(pipeline)
        self.completed = {
            stage: stage == RunStage.NOP for stage in [RunStage.NOP, *self.pipeline.stages, RunStage.DONE]
        }
        self.directories = {}
        self.target = target
        self.cache_hints = []
        self.config = config if config else {}
        self.features = features if features else []
        self.run_config = {}
        self.run_features = self.process_features(features)
        self.run_config = filter_config(self.config, "run", self.DEFAULTS, self.OPTIONAL, self.REQUIRED)
        self.sub_names = []
        self.sub_parents = {}
        self.times = {}
        self.failing = False  # -> RunStatus
        self.reason = None
        self.failed_stage = None
        # self.lock = threading.Lock()  # FIXME: use mutex instead of boolean
        self.locked = False
        self.report = None
        self.dir = None

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

    def has_target(self):
        return self.target is not None

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

    @property
    def postprocess_stage(self):
        return self.pipeline.postprocess_stage

    def has_stage(self, stage):
        """Select required instances and optional instances enabled by their implementation."""
        try:
            stage = self.pipeline.resolve(stage)
        except ValueError:
            return False
        if stage == RunStage.NOP:
            return True
        spec = self.pipeline.specs.get(stage)
        if spec is None:
            return False
        handler = STAGES.get(spec["uses"])
        return handler is None or not spec.get("optional", False) or handler().enabled(self)

    @property
    def completed_stages(self):
        return [stage for stage, completed in self.completed.items() if completed and stage != RunStage.NOP]

    @property
    def next_stage(self):
        """Determines the next not yet completed stage. Returns RunStage.DONE if already completed."""
        for stage, completed in self.completed.items():
            if not completed and self.has_stage(stage):
                return stage
        return RunStage.DONE

    @property
    def last_stage(self):
        """Determines the next not yet completed stage. Returns RunStage.DONE if already completed."""
        last = None
        # for stage in RunStage:
        for stage, completed in self.completed.items():
            if self.has_stage(stage):
                if not completed:
                    return last
                last = stage
        return last

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

    def init_directory(self, session=None, parent=None):
        """Initialize the temporary directory for this run."""
        if parent is not None:
            if not isinstance(parent, Path):
                assert isinstance(parent, str)
                parent = Path(parent)
            assert parent.is_dir()
            self.tempdir = None
            self.dir = parent / str(self.idx)
        elif session is not None:
            logger.warning(
                "session argument of run.init_directory is deprecated. Please use parent argument in the future."
            )
            self.tempdir = None
            self.dir = session.runs_dir / str(self.idx)
        else:
            assert not self.archived
            self.tempdir = tempfile.TemporaryDirectory()
            self.dir = Path(self.tempdir.name)
        self.dir.mkdir(exist_ok=True)
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

    def copy(self, session=None):
        """Create a new run based on this instance."""
        new = copy.deepcopy(self)
        assert session is not None, "Run.copy() needs session"
        if session:
            new_idx = session.request_run_idx()
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
        assert model is not None
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

    def get_unclean_components(self):
        ret = []
        if self.framework is not None:
            ret.append(self.framework.name)
        if self.frontends is not None and len(self.frontends) > 0:
            ret.extend([frontend.name for frontend in self.frontends])
        if self.backend is not None:
            ret.append(self.backend.name)
        if self.target is not None:
            ret.append(self.target.name)
        if self.platforms is not None and len(self.platforms) > 0:
            ret.extend([platform.name for platform in self.platforms])
        return list(set(ret))  # Eliminate duplicate names

    def check_unclean_components(self):
        unclean = self.get_unclean_components()
        if len(unclean) > 0:
            unclean_str = ", ".join(unclean)
            logger.warning("Added features after components leads to unexpected results: %s", unclean_str)

    def add_feature(self, feature, append=True):
        """Setter for a feature instance."""
        self.check_unclean_components()
        self.features = add_any(feature, self.features, append=append)
        self.run_features = self.process_features(self.features)

    def add_features(self, features, append=False):
        """Setter for the list of features."""
        if len(features) == 0:
            return
        self.check_unclean_components()
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
                raise ModelLookupError(f"No enabled frontend could load model '{model_name}'. Reasons: {reasons}")
            raise ModelLookupError(f"Model '{model_name}' was not found by any enabled frontend.")
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
                frontends_registry = get_frontends()
                assert name in frontends_registry, f"Unsupported frontend: {name}"
                frontends.append(self.init_component(frontends_registry[name], context=context))
            except Exception as e:
                reasons[name] = str(e)
                continue
        if len(frontends) == 0:
            if reasons:
                logger.error("Initialization of frontends was not successfull. Reasons: %s", reasons)
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
            self.add_backend(self.init_component(get_backends()[backend_name], context=context))
        if self.backend is None:
            return
        framework_name = self.backend.framework  # TODO: does this work?
        assert context.environment.has_framework(
            framework_name
        ), f"The framework '{framework_name}' is not enabled for this environment"
        self.add_framework(self.init_component(get_frameworks()[framework_name], context=context))

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
            postprocesses.append(self.init_component(get_postprocesses()[name], context=context))
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
        ret = sum(
            list(itertools.chain([subs[sub] for stage, subs in self.artifacts_per_stage.items() if sub in subs])), []
        )
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

    def stage_directory(self, stage, variant="default"):
        """Artifact directory, retaining numeric directories for legacy built-in IDs."""
        stage = self.pipeline.resolve(stage)
        dest = self.dir
        if self.stage_subdirs or isinstance(stage, StageId):
            label = stage.name.lower() if isinstance(stage, StageId) else str(int(stage))
            dest = dest / "stages" / label
        if variant not in ("", "default"):
            dest = dest / "sub" / variant
        return dest

    def export_stage(self, stage, optional=False):
        """Export stage artifacts of this run to its directory."""
        # TODO: per stage subdirs?
        if stage in self.artifacts_per_stage:
            for name in self.artifacts_per_stage[stage]:
                artifacts = self.artifacts_per_stage[stage][name]
                for artifact in artifacts:
                    if not artifact.optional or optional:
                        dest = self.stage_directory(stage, name)
                        dest.mkdir(parents=True, exist_ok=True)
                        extract = artifact.fmt in [ArtifactFormat.MLF, ArtifactFormat.ARCHIVE]
                        # extract = artifact.fmt == ArtifactFormat.MLF
                        # and not isinstance(self.platform, MicroTvmPlatform)
                        artifact.export(dest)
                        # Keep the tar as well as the extracted files
                        if extract:
                            artifact.export(dest, extract=True)

    # Compatibility entry points. Registered handlers own all stage logic;
    # these wrappers use the same engine as pipeline execution.
    def load(self):
        return self.execute_stage(self.pipeline.resolve(RunStage.LOAD))

    def tune(self):
        return self.execute_stage(self.pipeline.resolve(RunStage.TUNE))

    def build(self):
        return self.execute_stage(self.pipeline.resolve(RunStage.BUILD))

    def compile(self):
        return self.execute_stage(self.pipeline.resolve(RunStage.COMPILE))

    def run(self):
        return self.execute_stage(self.pipeline.resolve(RunStage.RUN))

    def postprocess(self):
        return self.execute_stage(self.pipeline.resolve(RunStage.POSTPROCESS))

    def invalidate_from(self, stage):
        for stale in self.pipeline.stages[self.pipeline.stages.index(stage) :]:
            self.completed[stale] = False
            self.artifacts_per_stage.pop(stale, None)
            self.times.pop(stale, None)
            self.sub_parents = {key: parent for key, parent in self.sub_parents.items() if key[0] != stale}
        self.sub_names = list(
            dict.fromkeys(name for variants in self.artifacts_per_stage.values() for name in variants)
        )
        self.report = None

    def execute_stage(self, stage):
        """Execute any registered stage and atomically commit its result."""
        stage = self.pipeline.resolve(stage)
        if stage in (RunStage.NOP, RunStage.DONE):
            raise ValueError("Pipeline sentinels are not executable stages")
        with self.resource_scope():
            self.lock()
            self.failing = False
            self.reason = None
            self.failed_stage = None
            try:
                self.invalidate_from(stage)
                return self._execute_stage(stage)
            except Exception as exc:
                self.invalidate_from(stage)
                self.failing = True
                self.reason = exc
                self.failed_stage = stage.name
                raise
            finally:
                self.unlock()

    def execute_custom_stage(self, stage):
        """Compatibility alias for the common stage engine."""
        return self.execute_stage(stage)

    def _execute_stage(self, stage):
        spec = self.pipeline.specs[stage]
        implementation = STAGES.get(spec["uses"])
        if implementation is None:
            raise ValueError(f"Unknown stage implementation: {spec['uses']}. Load its extensions.py in every worker.")
        started = time.time()
        handler = implementation()
        # Capability validation precedes default input resolution, so a required
        # stage cannot silently disappear or fail with a misleading input error.
        if not handler.available(self):
            raise ValueError(f"Stage {stage.name} requires unavailable component capabilities")
        if self.dir is None:
            self.init_directory()
        workdir = self.dir / "stages" / stage.name.lower()
        workdir.mkdir(parents=True, exist_ok=True)
        spec = self.pipeline.stage_spec(self, stage)
        inputs = self.pipeline.inputs(self, stage, spec)
        for binding in spec["inputs"].values():
            self.export_stage(stage_key(binding.split(".", 1)[0]), optional=True)
        context = StageContext(self, spec, inputs, workdir)
        handler.validate(context)
        result = handler.execute(context)
        if not isinstance(result, StageResult):
            raise TypeError("Stages must return StageResult")
        sources = {stage_key(binding.split(".", 1)[0]) for binding in spec["inputs"].values()}
        parents = {}
        artifacts = {name: list(items) for name, items in result.artifacts.items()}
        # Component APIs still emit implementation-named metrics. Give every
        # instance its own filename without mutating the component's artifact.
        for name, items in artifacts.items():
            for index, artifact in enumerate(items):
                if (
                    isinstance(artifact, Artifact)
                    and artifact.name == f"{spec['uses']}_metrics.csv"
                    and stage.name.lower() != spec["uses"]
                ):
                    artifact = copy.copy(artifact)
                    artifact.name = f"{stage.name.lower()}_metrics.csv"
                    if artifact.content is not None:
                        artifact.path = None
                    items[index] = artifact
        if set(result.parents) - set(artifacts) or set(result.metrics) - set(artifacts):
            raise ValueError("Parents and metrics must refer to output variants")
        for name in artifacts:
            if not name or Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Invalid output variant: {name}")
            if not all(isinstance(artifact, Artifact) for artifact in artifacts[name]):
                raise TypeError("Stage outputs must contain Artifact objects")
            parent = result.parents.get(name)
            if parent is None and sources:
                raise ValueError(f"Missing parent for output variant: {name}")
            if parent is not None:
                parent = (stage_key(parent[0]), parent[1])
                # RUN's deployment binding, for example, preserves flash logs.
                if parent[0] not in sources or parent[1] not in self.artifacts_per_stage.get(parent[0], {}):
                    raise ValueError(f"Invalid output parent: {parent}")
            parents[(stage, name)] = parent if parent else (None, None)
            if name in result.metrics:
                if lookup_artifacts(artifacts[name], name=f"{stage.name.lower()}_metrics.csv"):
                    raise ValueError("Return either a metrics CSV or StageResult.metrics, not both")
                metrics = Metrics()
                for key, value in result.metrics[name].items():
                    metrics.add(key, value)
                artifacts[name].append(
                    Artifact(f"{stage.name.lower()}_metrics.csv", content=metrics.to_csv(), fmt=ArtifactFormat.TEXT)
                )
        self.artifacts_per_stage[stage] = artifacts
        self.sub_parents.update(parents)
        self.sub_names = list(dict.fromkeys([*self.sub_names, *artifacts]))
        self.completed[stage] = True
        if self.profile_stages:
            self.times[stage] = (started, time.time())
        if result.report is not None:
            self.report = result.report
        self.export_stage(stage, optional=self.export_optional)
        return result

    @contextmanager
    def resource_scope(self):
        """Keep declared resources and device state scoped to one invocation."""
        from contextlib import ExitStack
        import hashlib
        from filelock import FileLock

        if getattr(self, "_resource_locks_held", False):
            yield
            return
        resources = sorted(
            {
                resource
                for spec in self.pipeline.specs.values()
                for resource in spec.get("resources", {}).get("exclusive", [])
            }
        )
        self._flashed_program = None
        self._flash_parent = None
        with ExitStack() as stack:
            for resource in resources:
                digest = hashlib.sha256(resource.encode()).hexdigest()
                stack.enter_context(FileLock(Path(tempfile.gettempdir()) / f"mlonmcu-resource-{digest}.lock"))
            self._resource_locks_held = True
            try:
                yield
            finally:
                self._resource_locks_held = False
                self._flashed_program = None
                self._flash_parent = None

    def process(self, until=RunStage.RUN, start=None, skip=None, export=False):
        """Execute an inclusive pipeline prefix through the common stage engine."""
        with self.resource_scope():
            return self._process_pipeline(until, start, skip, export)

    def _process_pipeline(self, until, start, skip, export):
        skip_keys = set()
        for value in skip or []:
            try:
                skip_keys.add(self.pipeline.resolve(value))
            except ValueError:
                skip_keys.add(stage_key(value))
        skip = skip_keys
        stages = self.pipeline.through(until)
        explicit_start = start is not None
        start = self.next_stage if start is None else self.pipeline.resolve(start)
        if explicit_start and start in self.pipeline.stages and start in stages:
            self.invalidate_from(start)
        if start == RunStage.DONE:
            stages = []
        elif start != RunStage.NOP:
            stages = stages[stages.index(start) :] if start in stages else []
        for stage in stages:
            if not self.has_stage(stage) or stage in skip:
                continue
            self.failing = False
            self.failed_stage = None
            self.reason = None
            try:
                self.execute_stage(stage)
            except Exception as exc:
                self.failing = True
                self.reason = exc
                self.failed_stage = stage.name
                logger.exception("%s Run failed at stage '%s'", self.prefix, stage.name)
                break
        report = self.get_report()
        if export:
            self.export(optional=self.export_optional)
            report.export(Path(self.dir) / "report.csv")
        return report

    def write_run_file(self):
        """Create a run.txt file which contains information used to reconstruct the run based
        on its properties at a later point in time."""
        logger.debug("%s Writing run file", self.prefix)
        filename = self.dir / "run.txt"
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(str(self))
        with open(self.dir / "pipeline.yml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.pipeline.serialize(), handle, sort_keys=False)

    @property
    def prefix(self):
        """Get prefix property."""
        session = None  # TODO: fix
        return (
            (f"[session-{session.idx}] [run-{self.idx}]" if session else f"[run-{self.idx}]")
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
        if self.model:
            ret.update(config_helper(self.model))
        for frontend in self.frontends:
            ret.update(config_helper(frontend))
        if self.backend:
            self.backend.reconfigure()
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

    def get_report(self, session=None):
        """Returns teh complete report of this run."""
        if self.completed.get(self.postprocess_stage, False):
            if self.report is not None:
                return (
                    self.report
                )  # Use postprocessed report instead of generating a new one (TODO: find a better approach)
        # TODO: config or args for stuff like (session id) and run id as well as detailed features and configs
        report = Report()
        pre = {}
        if session is not None:
            pre["Session"] = session.idx
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
        # post["Config"] = self.get_all_configs(omit_paths=True, omit_defaults=False, omit_globals=True)
        post["Config"] = self.get_all_configs(omit_paths=False, omit_defaults=False, omit_globals=True)
        post["Postprocesses"] = self.get_all_postprocess_names()
        post["Stages"] = [stage.name for stage in self.completed_stages]
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

        metric_cache = {}

        def variant_metrics(stage, name):
            key = (stage, name)
            if key in metric_cache:
                return metric_cache[key]
            metrics = Metrics()
            parent_stage, parent_name = self.sub_parents.get(key, (None, None))
            if parent_stage is not None:
                inherited = variant_metrics(parent_stage, parent_name)
                for metric, value in inherited.get_data(include_optional=self.export_optional).items():
                    metrics.add(metric, value)
            artifacts = self.artifacts_per_stage[stage][name]
            matches = lookup_artifacts(artifacts, name=f"{stage.name.lower()}_metrics.csv")
            if matches:
                assert len(matches) == 1
                current = Metrics.from_csv(matches[0].content)
                for metric, value in current.get_data(include_optional=self.export_optional).items():
                    metrics.add(metric, value, overwrite=True)
            metric_cache[key] = metrics
            return metrics

        for stage in self.pipeline.stages:
            if stage == self.postprocess_stage or not self.artifacts_per_stage.get(stage):
                continue
            subs = list(self.artifacts_per_stage[stage])
            metrics_by_sub = {name: variant_metrics(stage, name) for name in subs}
        if not subs:
            subs = ["default"]

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
            if self.profile_stages:
                for stage, stage_times in self.times.items():
                    assert len(stage_times) == 2
                    start, end = stage_times
                    main[f"{stage_key(stage).name.capitalize()} Stage Start Time [s]"] = start
                    main[f"{stage_key(stage).name.capitalize()} Stage End Time [s]"] = end
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
        for stage in self.completed_stages:
            if not self.has_stage(stage):
                continue
            self.export_stage(stage, optional=optional)

        self.write_run_file()

    def result(self):
        return RunResult(self)  # TODO: session?

    # Everything in Run is serializable except PlatformTargets and PlatformBackends...
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     print("state", state)
    #     # Don't pickle baz
    #     # del state["baz"]
    #     # del state["platforms"]
    #     del state["target"]
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # Add baz back since it doesn't exist in the pickle
    #     # self.baz = 0

    def save(self, dest: Union[str, Path], fmt: Optional[str] = None):
        raise NotImplementedError
        # if not isinstance(dest, Path):
        #     assert isinstance(dest, str)
        #     dest = Path(dest)
        # # data = self._serialize()
        # if fmt is None:
        #     fmt = dest.suffix
        #     assert fmt
        #     if fmt[0] == ".":
        #         fmt = fmt[1:]
        # if fmt.lower() in ["pkl", "pickle"]:
        #     import pickle
        #     with open(dest, "wb") as f:
        #         pickle.dump(self, f)
        # else:
        #     raise ValueError(f"Unsupported format: {fmt}")

    def save_artifacts(self, dest: Union[str, Path], fmt: Optional[str] = None, full: bool = False):
        if not isinstance(dest, Path):
            assert isinstance(dest, str)
            dest = Path(dest)
        artifacts = self.artifacts
        if fmt is None:
            fmt = dest.suffix
            assert fmt
            if fmt[0] == ".":
                fmt = fmt[1:]
        if fmt.lower() in ["pkl", "pickle"]:
            data = {"artifacts": artifacts}
            import pickle

            with open(dest, "wb") as f:
                pickle.dump(data, f)
        elif fmt.lower() in ["yml", "yaml"]:
            data = {"artifacts": [artifact.serialize(full=full) for artifact in artifacts]}

            with open(dest, "w") as f:
                yaml.dump(data, f, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def cleanup_artifacts(self, dirs: bool = False):
        artifacts = self.artifacts
        for artifact in artifacts:
            if artifact.fmt in [ArtifactFormat.ARCHIVE]:
                logger.warning("Can not cleanup unzipped archives!")
            if not artifact.exported:
                continue
            assert artifact.path.is_absolute()
            if artifact.path.is_file():
                artifact.path.unlink()
            elif artifact.path.is_dir() and dirs:
                shutil.rmtree(artifact.path)
            else:
                assert False

    def cleanup_directories(self):
        for directory in self.directories.values():
            assert directory.is_absolute()
            if not directory.is_dir():
                continue
            shutil.rmtree(directory)

    def initializer(self):
        return RunInitializer(
            idx=self.idx,
            model_name=self.model.name if self.model else None,
            framework_name=self.framework.name if self.framework else None,
            frontend_names=[frontend.name for frontend in self.frontends],
            backend_name=self.backend.name if self.backend else None,
            target_name=self.target.name if self.target else None,
            platform_names=[platform.name for platform in self.platforms],
            feature_names=[feature.name for feature in self.features],
            config={**self.config},
            postprocess_names=[postprocess.name for postprocess in self.postprocesses],
            comment=self.comment,
            pipeline=self.pipeline.serialize(),
        )


class ArchivedRun(Run):

    def __init__(self, **kwargs):
        super().__init__(
            self,
            archived=True,
            **kwargs,
        )

    @staticmethod
    def from_file(path: Union[Path, str]):
        path = Path(path)
        assert path.is_file()
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)
        print("yaml_data", yaml_data)
        return ArchivedRun(**yaml_data)

    @staticmethod
    def from_dir(path: Union[Path, str], allow_missing: bool = True):
        path = Path(path)
        assert path.is_dir()
        run_yaml = path / "run.yml"
        if not run_yaml.is_file():
            assert allow_missing, f"Run YAML does not exist: {path}"
            ret = ArchivedRun()
            ret.dir = path
            return ret
        return ArchivedRun.from_file(run_yaml, allow_missing=allow_missing)

    @staticmethod
    def restore(self):
        raise NotImplementedError
