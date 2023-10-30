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
"""Definitions of a task registry used to automatically install dependencies."""

from functools import wraps
import itertools
from enum import Enum
import time
from typing import List, Tuple
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from tqdm import tqdm

from mlonmcu.logging import get_logger

logger = get_logger()


def get_combs(data) -> List[dict]:
    """Utility which returns combinations of the input data.

    Parameters
    ----------
    data : dict
        Input dictionary

    Returns
    -------
    combs : list
        All combinations of the input data.

    Examples
    --------
    >>> get_combs({"foo": [False, True], "bar": [5, 10]})
    [{"foo": False, "bar": 5}, {"foo": False, "bar": 10}, {"foo": True, "bar": 5}, {"foo": True, "bar": 10}]
    """
    keys = list(data.keys())
    values = list(data.values())
    prod = list(itertools.product(*values))
    if len(prod) == 1:
        if len(prod[0]) == 0:
            prod = []
    combs = [dict(zip(keys, p)) for p in prod]
    return combs


class TaskType(Enum):
    """Enumeration for the task type."""

    MISC = 0
    FRAMEWORK = 1
    BACKEND = 2
    TOOLCHAIN = 3
    TARGET = 4
    FRONTEND = 5
    OPT = 6
    FEATURE = 7
    PLATFORM = 8


class TaskGraph:
    """Task graph object.

    Attributes
    ----------
    names : list
        list of task names in the graph
    dependencies : dict
        Dependencies between task artifacts
    providers : dict
        Providers for all the artifacts

    Examples
    -------
    TODO
    """

    def __init__(self, names: List[str], dependencies: dict, providers: dict):
        self.names = names
        self.dependencies = dependencies
        self.providers = providers

    def get_graph(self) -> Tuple[list, list]:
        """Get nodes and edges of the task graph.

        Returns
        -------
        nodes : list
            List of edges
        edges : list
            List of edge tuples.
        """
        nodes = list(self.names)
        edges = []
        for dest, deps in self.dependencies.items():
            for dep in deps:
                if dep not in self.providers.keys():
                    raise RuntimeError(f"Unable to resolve dependency '{dep}'")
                src = self.providers[dep]
                edge = (src, dest)
                edges.append(edge)
        # Remove duplicates
        edges = list(dict.fromkeys(edges))
        return nodes, edges

    def get_order(self) -> list:
        """Get execution order of tasks via topological sorting."""
        nodes, edges = self.get_graph()
        graph = nx.DiGraph(edges)
        graph.add_nodes_from(nodes)
        order = list(nx.topological_sort(graph))
        return order

    def export_dot(self, path):
        """Visualize the task dependency graph."""
        nodes, edges = self.get_graph()
        graph = nx.DiGraph(edges)
        graph.add_nodes_from(nodes)
        # order = list(nx.topological_sort(graph))
        # TODO: annotate with order
        # TODO: also export order as extra graph
        write_dot(graph, path)


class TaskFactory:
    """Class which is used to register all available tasks and their annotations.

    Attributes
    ----------
    registry : dict
        Mapping of task names and their actual function
    dependencies : dict
        Mapping of task dependencies
    providers : dict
        Mapping of which task provides which artifacts
    types : dict
        Mapping of task types
    validates : dict
        Mapping of validation functions for the tasks
    changed : list
        List of tasks?artifacts which have changed recently
    """

    def __init__(self):
        self.registry = {}
        self.dependencies = {}
        self.providers = {}
        self.types = {}
        self.params = {}
        self.validates = {}
        self.changed = []  # Main problem: per

    def reset_changes(self):
        """Reset all pending changes."""
        self.changed = []

    def needs(self, keys, force=True):
        """Decorator which registers the artifacts a task needs to be processed."""

        def real_decorator(function):
            name = function.__name__
            if name in self.dependencies:
                self.dependencies[name].extend(keys)
            else:
                self.dependencies[name] = keys

            @wraps(function)
            def wrapper(*args, **kwargs):
                # logger.debug("Checking inputs...")
                if force:
                    context = args[0]
                    variables = context.cache._vars
                    for key in keys:
                        if key not in variables.keys() or variables[key] is None:
                            raise RuntimeError(f"Task '{name}' needs the value of '{key}' which is not set")
                retval = function(*args, **kwargs)
                return retval

            return wrapper

        return real_decorator

    def optional(self, keys):
        """Decorator for optional task requirements."""
        return self.needs(keys, force=False)

    def removes(self, keys):
        """Decorator for cleanuo tasks."""

        # TODO: implementation
        def real_decorator(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                retval = function(*args, **kwargs)
                return retval

            return wrapper

        return real_decorator

    # def optional(self, keys):
    #     def real_decorator(function):
    #         name = function.__name__
    #         if name in self.dependencies:
    #             self.dependencies[name].extend(keys)
    #         else:
    #             self.dependencies[name] = keys
    #         @wraps(function)
    #         def wrapper(*args, **kwargs):
    #             retval = function(*args, **kwargs)
    #             return retval
    #         return wrapper
    #     return real_decorator

    def provides(self, keys):
        """Decorator which registers what a task provides."""

        def real_decorator(function):
            name = function.__name__
            for key in keys:
                self.providers[key] = name

            @wraps(function)
            def wrapper(*args, **kwargs):
                context = args[0]
                for key in keys:
                    if key in context.cache._vars:
                        del context.cache._vars[key]  # Unset the value before calling function
                retval = function(*args, **kwargs)
                if retval is not False:
                    # logger.debug("Checking outputs...")
                    variables = context.cache._vars
                    for key in keys:
                        if key not in variables.keys() or variables[key] is None:
                            raise RuntimeError(f"Task '{name}' did not set the value of '{key}'")
                return retval

            self.registry[name] = wrapper
            return wrapper

        return real_decorator

    def param(self, flag, options):
        """Decorator which registers available task parameters."""
        if not isinstance(options, list):
            options = [options]

        def real_decorator(function):
            name = function.__name__
            if name in self.params:
                self.params[name][flag] = options
            else:
                self.params[name] = {flag: options}

            @wraps(function)
            def wrapper(*args, **kwargs):
                retval = function(*args, **kwargs)
                return retval

            return wrapper

        return real_decorator

    def validate(self, func):
        """Decorator which registers validation functions for a task."""

        def real_decorator(function):
            name = function.__name__
            self.validates[name] = func
            return function

        return real_decorator

    def register(self, category=TaskType.MISC):
        """Decorator which actually registers a task in the registry."""

        def real_decorator(function):
            name = function.__name__

            @wraps(function)
            def wrapper(*args, rebuild=False, progress=False, **kwargs):
                combs = get_combs(self.params[name])

                def get_valid_combs(combs):
                    ret = []
                    for comb in combs:
                        if name in self.validates:
                            check = self.validates[name](args[0], params=comb)
                            if not check:
                                continue
                        ret.append(comb)
                    return ret

                combs_ = get_valid_combs(combs)

                def process(name_, params=None, rebuild=False):
                    if not params:
                        params = []
                    rebuild = rebuild
                    if name in self.dependencies:
                        for dep in self.dependencies[name]:
                            if dep in self.changed:
                                rebuild = True
                                break
                    retval = function(*args, params=params, rebuild=rebuild, **kwargs)
                    if retval:
                        keys = [key for key, provider in self.providers.items() if provider == name]
                        for key in keys:
                            if key not in self.changed:
                                self.changed.append(key)
                    # logger.debug("Processed task:", function.__name__)
                    return retval

                if progress:
                    pbar = tqdm(
                        total=max(len(combs_), 1),
                        desc="Processing",
                        ncols=100,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}s]",
                        leave=None,
                    )
                else:
                    pbar = None
                if len(combs_) == 0:
                    if pbar:
                        pbar.set_description(f"Processing: {name}")
                    else:
                        logger.info("Processing task: %s", name)
                    time.sleep(0.1)
                    check = True
                    if len(combs) > 0:
                        check = False
                    else:
                        if name in self.validates:
                            check = self.validates[name](args[0], params={})
                    if check:
                        start = time.time()
                        retval = process(name, rebuild=rebuild)
                        end = time.time()
                        diff = end - start
                        minutes = int(diff // 60)
                        seconds = int(diff % 60)
                        duration_str = f"{seconds}s" if minutes == 0 else f"{minutes}m{seconds}s"
                        if not pbar:
                            logger.debug("-> Done (%s)", duration_str)
                        # TODO: move this to helper func
                    else:
                        logger.debug("-> Skipped")
                        retval = False
                    if pbar:
                        pbar.update(1)
                else:
                    for comb in combs_:  # TODO process in parallel?
                        extended_name = name + str(comb)
                        if pbar:
                            pbar.set_description(f"Processing - {extended_name}")
                        else:
                            logger.info("Processing task: %s", extended_name)
                        time.sleep(0.1)
                        start = time.time()
                        retval = process(extended_name, params=comb, rebuild=rebuild)
                        end = time.time()
                        diff = end - start
                        minutes = int(diff // 60)
                        seconds = int(diff % 60)
                        duration_str = f"{seconds}s" if minutes == 0 else f"{minutes}m{seconds}s"
                        if not pbar:
                            logger.debug("-> Done (%s)", duration_str)
                            # TODO: move this to helper func
                        else:
                            pbar.update(1)
                if pbar:
                    pbar.close()
                return retval

            self.registry[name] = wrapper
            self.types[name] = category
            self.params[name] = {}
            return wrapper

        return real_decorator
