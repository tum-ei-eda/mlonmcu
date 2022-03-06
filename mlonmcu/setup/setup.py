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
import time
from tqdm import tqdm

from mlonmcu.logging import get_logger
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import get_matching_features
from mlonmcu.config import filter_config
from .tasks import Tasks
from .task import TaskGraph
from mlonmcu.utils import ask_user

logger = get_logger()


class Setup:
    """MLonMCU dependency management interface."""

    FEATURES = []

    DEFAULTS = {
        "print_outputs": False,
    }

    REQUIRED = []

    def __init__(self, features=None, config=None, context=None, tasks_factory=Tasks):
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, "setup", self.DEFAULTS, self.REQUIRED)
        self.context = context
        self.tasks_factory = tasks_factory
        self.verbose = bool(self.config["print_outputs"])

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
                shutil.rmtree(full_path)
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
            feature.add_setup_config(self.config)
        return features

    def get_dependency_order(self):
        self.tasks_factory.reset_changes()
        task_graph = TaskGraph(
            self.tasks_factory.registry.keys(),
            self.tasks_factory.dependencies,
            self.tasks_factory.providers,
        )
        V, E = task_graph.get_graph()
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

    def invoke_single_task(self, name, progress=False, write_cache=True, rebuild=False):
        assert name in self.tasks_factory.registry, f"Invalid task name: {name}"
        func = self.tasks_factory.registry[name]
        func(self.context, progress=progress, rebuild=rebuild, verbose=self.verbose)

    def install_dependencies(
        self,
        progress=False,
        write_cache=True,
        rebuild=False,
    ):
        assert self.context is not None
        order = self.get_dependency_order()
        pbar = self.setup_progress_bar(progress)
        for task in order:
            func = self.tasks_factory.registry[task]
            func(self.context, progress=progress, rebuild=rebuild, verbose=self.verbose)
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
        if write_cache:
            self.write_cache_file()
        logger.info("Finished installing dependencies")
