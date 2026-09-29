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
import os
import shutil
import multiprocessing
from tqdm import tqdm

from mlonmcu.logging import get_logger
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import get_matching_features
from mlonmcu.config import filter_config, str2bool
from .tasks import get_task_factory
from .task import TaskGraph
from mlonmcu.utils import ask_user

logger = get_logger()


class Setup:
    """MLonMCU dependency management interface."""

    FEATURES = set()

    DEFAULTS = {
        "print_outputs": False,
        "num_threads": None,
    }

    REQUIRED = set()
    OPTIONAL = set()

    def __init__(self, features=None, config=None, context=None, tasks_factory=None):
        if not tasks_factory:
            tasks_factory = get_task_factory()
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.context = context
        if self.context is not None:
            self.context.environment.vars.update(self.config)
        self.config = filter_config(self.config, "setup", self.DEFAULTS, self.OPTIONAL, self.REQUIRED)
        self.tasks_factory = tasks_factory
        self.num_threads = int(
            self.config["num_threads"] if self.config["num_threads"] else multiprocessing.cpu_count()
        )

    @property
    def verbose(self):
        value = self.config["print_outputs"]
        return str2bool(value)

    def clean_cache(self, interactive=True):
        assert self.context is not None
        deps_dir = self.context.environment.lookup_path("deps").path
        cache_file = deps_dir / "cache.ini"
        if cache_file.is_file():
            print(f"The dependency cache file ({cache_file}) will be removed.")
            if ask_user("Are your sure?", default=not interactive, interactive=interactive):
                print(f"Removing {cache_file} ...")
                os.remove(cache_file)

    def clean_dependencies(self, interactive=True):
        assert self.context is not None
        self.clean_cache(interactive=interactive)
        deps_dir = self.context.environment.lookup_path("deps").path
        subdirs = ["src", "build", "install"]
        print(f"All dependencies will be removed from {deps_dir}.")
        if ask_user("Are your sure?", default=not interactive, interactive=interactive):
            for subdir in subdirs:
                full_path = deps_dir / subdir
                print(f"Removing contents of {full_path} ...")
                shutil.rmtree(full_path, ignore_errors=True)
                full_path.mkdir(exist_ok=True)

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.SETUP)
        for feature in features:
            # Not need to list features explicitly
            # assert (
            #     feature.name in self.FEATURES
            # ), f"Incompatible feature: {feature.name}"
            feature.used = True
            feature.add_setup_config(self.config)
        return features

    def _get_task_graph(self):
        self.tasks_factory.reset_changes()
        task_graph = TaskGraph(
            self.tasks_factory.registry.keys(),
            self.tasks_factory.dependencies,
            self.tasks_factory.providers,
        )
        return task_graph

    def get_dependency_order(self):
        task_graph = self._get_task_graph()
        # V, E = task_graph.get_graph()
        order = task_graph.get_order()
        order_str = " -> ".join(order)
        logger.debug("Determined dependency order: %s" % order_str)
        return order

    def setup_progress_bar(self, enabled):
        if enabled:
            pbar = tqdm(
                total=len(self.tasks_factory.registry),
                desc="Installing dependencies",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}s]",
            )
            return pbar
        else:
            logger.info("Installing dependencies...")
            return None

    def write_cache_file(self):
        logger.debug("Updating dependency cache")
        cache_file = self.context.environment.paths["deps"].path / "cache.ini"
        self.context.cache.write_to_file(cache_file)

    def write_env_file(self):
        logger.debug("Updating paths cript")
        paths_file = self.context.environment.paths["deps"].path / "paths.sh"
        paths = self.context.export_paths
        temp = ":".join(map(str, paths))
        content = f"export PATH={temp}:$PATH"
        with open(paths_file, "w") as handle:
            handle.write(content)

    def visualize(self, path, ordered=False):
        task_graph = self._get_task_graph()
        task_graph.export_dot(path)
        logger.debug("Written task graph to file: %s" % path)

    def invoke_single_task(self, name, progress=False, write_cache=True, write_env=True, rebuild=False):
        assert name in self.tasks_factory.registry, f"Invalid task name: {name}"
        func = self.tasks_factory.registry[name]
        func(self.context, progress=progress, rebuild=rebuild, verbose=self.verbose, threads=self.num_threads)
        if write_cache:
            self.write_cache_file()
        if write_env:
            self.write_env_file()

    def install_dependencies(
        self,
        progress=False,
        write_cache=True,
        write_env=True,
        rebuild=False,
    ):
        assert self.context is not None
        order = self.get_dependency_order()
        pbar = self.setup_progress_bar(progress)
        for task in order:
            func = self.tasks_factory.registry[task]
            func(self.context, progress=progress, rebuild=rebuild, verbose=self.verbose, threads=self.num_threads)
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
        if write_cache:
            self.write_cache_file()
        if write_env:
            self.write_env_file()
        logger.info("Finished installing dependencies")
        return True

    def generate_requirements(
        self,
    ):
        from .gen_requirements import get_extra_requirements, join_requirements, merge_requirements
        import os

        requirements = join_requirements()
        output_dir = os.path.join(self.context.environment.home, "requirements_addition.txt")
        selected_requirements = []
        additional_lines = []

        def feature_enabled_and_supported(obj, feature):
            if getattr(obj, "enabled", True):
                if getattr(obj, "name", None) == feature and getattr(obj, "supported", False):
                    return True
                else:
                    enable_and_supported = False
                    for k, v in obj.__dict__.items():
                        if isinstance(v, list):
                            for i in v:
                                enable_and_supported = enable_and_supported or feature_enabled_and_supported(i, feature)
                    return enable_and_supported
            else:
                return False

        def add_requirements(piece):
            selected_requirements.extend(requirements[piece][1])

        config_pools = (
            self.context.environment.frameworks
            + self.context.environment.frontends
            + self.context.environment.platforms
            + self.context.environment.targets
        )

        def has_enabled(name):
            return any(name in config.name and config.enabled for config in config_pools)

        for name, piece in [
            ("zephyr", "zephyr"),
            ("tflm", "tflm"),
            ("tflite", "tflite"),
            ("onnx", "onnx"),
            ("etiss", "etiss"),
            ("microtvm_gvsoc", "microtvm_gvsoc"),
            ("gvsoc_pulp", "gvsoc_pulp"),
            ("visualize", "visualize"),
            ("microtvm", "microtvm"),
            ("tvm", "tvm"),
        ]:
            if has_enabled(name):
                add_requirements(piece)
                logger.info("add dependencies for %s", piece)
        if any(feature_enabled_and_supported(config, "moiopt") for config in config_pools):
            add_requirements("moiopt")
        if any("tvm" in config.name and feature_enabled_and_supported(config, "autotuned") for config in config_pools):
            add_requirements("tvm-autotuning")
        if any("tf" in config.name and feature_enabled_and_supported(config, "visualize") for config in config_pools):
            add_requirements("visualize_tflite")
        if has_enabled("iree"):
            add_requirements("iree")
            vars = self.context.environment.vars
            cache = self.context.cache
            iree_build_dir = vars.get("iree.build_dir") or cache.get("iree.build_dir")
            iree_src_dir = vars.get("iree.src_dir") or cache.get("iree.src_dir")
            if iree_src_dir is not None:
                additional_lines.append(f"-r {iree_src_dir}/integrations/tensorflow/test/requirements.txt")
                additional_lines.append(
                    f"-r {iree_src_dir}/runtime/bindings/python/iree/runtime/build_requirements.txt"
                )
            if iree_build_dir is not None:
                additional_lines.append(f"-e {iree_build_dir}/compiler")
                additional_lines.append(f"-e {iree_build_dir}/runtime")

        python = self.context.environment.python
        selected_requirements.extend(get_extra_requirements(python.extra))
        selected_requirements.extend(python.requirements)
        selected_requirements = merge_requirements(selected_requirements)
        with open(output_dir, "w") as f:
            f.write(f"# AUTOGENERATED by 'mlonmcu setup -g' {os.linesep}")
            for requirement in selected_requirements + additional_lines:
                f.write(f"{requirement}{os.linesep}")
        logger.info("Finished generating requirements_addition.txt")
        return True
