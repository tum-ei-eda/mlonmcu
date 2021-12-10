from functools import wraps
import itertools
from enum import Enum
import networkx as nx
import logging

logger = logging.getLogger('mlonmcu')
logger.setLevel(logging.DEBUG)

class TaskType(Enum):
    MISC = 0
    FRAMEWORK = 1
    BACKEND = 2
    TOOLCHAIN = 3
    TARGET = 4
    FRONTEND = 5
    OPT = 6


class Task:

    registry = {}
    dependencies = {}
    providers = {}
    types = {}
    params = {}
    validates = {}
    changed = []  # Main problem: per

    @staticmethod
    def reset_changes():
        Task.changed = []

    @staticmethod
    def get_graph():
        nodes = list(Task.registry.keys())
        edges = []
        for dest, deps in Task.dependencies.items():
            for dep in deps:
                if dep not in Task.providers.keys():
                    raise RuntimeError(f"Unable to resolve dependency '{dep}'")
                src = Task.providers[dep]
                edge = (src, dest)
                edges.append(edge)
        # Remove duplicates
        edges = list(dict.fromkeys(edges))
        return nodes, edges

    @staticmethod
    def get_order():
       V, E = Task.get_graph()
       G = nx.DiGraph(E)
       G.add_nodes_from(V)
       order = list(nx.topological_sort(G))
       return order

    @staticmethod
    def needs(keys, force=True):
        def real_decorator(function):
            name = function.__name__
            if name in Task.dependencies:
                Task.dependencies[name].extend(keys)
            else:
                Task.dependencies[name] = keys
            @wraps(function)
            def wrapper(*args, **kwargs):
                # logger.debug("Checking inputs...")
                if force:
                    context = args[0]
                    variables = context._vars
                    for key in keys:
                        if key not in variables.keys() or variables[key] is None:
                            raise RuntimeError(f"Task '{name}' needs the value of '{key}' which is not set")
                retval = function(*args, **kwargs)
                return retval
            return wrapper
        return real_decorator

    @staticmethod
    def optional(keys):
        return Task.needs(keys, force=False)

    # @staticmethod
    # def optional(keys):
    #     def real_decorator(function):
    #         name = function.__name__
    #         if name in Task.dependencies:
    #             Task.dependencies[name].extend(keys)
    #         else:
    #             Task.dependencies[name] = keys
    #         @wraps(function)
    #         def wrapper(*args, **kwargs):
    #             retval = function(*args, **kwargs)
    #             return retval
    #         return wrapper
    #     return real_decorator

    @staticmethod
    def provides(keys):
        def real_decorator(function):
            name = function.__name__
            for key in keys:
                Task.providers[key] = name
            @wraps(function)
            def wrapper(*args, **kwargs):
                context = args[0]
                for key in keys:
                    if key in context._vars:
                        del context._vars[key]  # Unset the value before calling function
                retval = function(*args, **kwargs)
                # logger.debug("Checking outputs...")
                variables = context._vars
                for key in keys:
                    if key not in variables.keys() or variables[key] is None:
                        raise RuntimeError(f"Task '{name}' did not set the value of '{key}'")
                return retval
            Task.registry[name] = wrapper
            return wrapper
        return real_decorator

    @staticmethod
    def param(flag, options):
        if not isinstance(options, list):
            options = [options]
        def real_decorator(function):
            name = function.__name__
            if name in Task.params:
                Task.params[name][flag] = options
            else:
                Task.params[name] = {flag: options}
            @wraps(function)
            def wrapper(*args, **kwargs):
                retval = function(*args, **kwargs)
                return retval
            return wrapper
        return real_decorator

    @staticmethod
    def validate(func):
        def real_decorator(function):
            name = function.__name__
            Task.validates[name] = func
            return function
        return real_decorator

    def get_combs(data):
        keys = list(data.keys())
        values = list(data.values())
        prod = list(itertools.product(*values))
        if len(prod) == 1:
            if len(prod[0]) == 0:
                prod = []
        combs = [dict(zip(keys, p)) for p in prod]
        return combs

    @staticmethod
    def register(category=TaskType.MISC):
        def real_decorator(function):
            name = function.__name__
            @wraps(function)
            def wrapper(*args, **kwargs):
                combs = Task.get_combs(Task.params[name])
                def process(name_, params={}):
                    if name in Task.validates:
                        check = Task.validates[name](args[0], params=comb)
                        if not check:
                            # logger.debug("Skipping task: %s", name_)
                            return
                    rebuild = False
                    if name in Task.dependencies:
                        for dep in Task.dependencies[name]:
                            if dep in Task.changed:
                                rebuild = True
                                break
                    logger.info("Processing task: %s", name_)
                    retval = function(*args, params=params, rebuild=rebuild, **kwargs)
                    if retval:
                        keys = [key for key, provider in Task.providers.items() if provider == name]
                        for key in keys:
                            if key not in Task.changed:
                                Task.changed.append(key)
                    # logger.debug("Processed task:", function.__name__)
                    return retval

                if len(combs) == 0:
                    retval = process(name)
                else:
                    for comb in combs:
                        extended_name = name + str(comb)
                        retval = process(extended_name, params=comb)
                return retval
            Task.registry[name] = wrapper
            Task.types[name] = category
            Task.params[name] = {}
            return wrapper
        return real_decorator
