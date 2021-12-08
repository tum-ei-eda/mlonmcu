from functools import wraps
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


class Task:

    registry = {}
    dependencies = {}
    providers = {}
    types = {}

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
    def needs(keys):
        def real_decorator(function):
            name = function.__name__
            Task.dependencies[name] = keys
            @wraps(function)
            def wrapper(*args, **kwargs):
                # logger.debug("Checking inputs...")
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
    def provides(keys):
        def real_decorator(function):
            name = function.__name__
            for key in keys:
                Task.providers[key] = name
            @wraps(function)
            def wrapper(*args, **kwargs):
                retval = function(*args, **kwargs)
                # logger.debug("Checking outputs...")
                context = args[0]
                variables = context._vars
                for key in keys:
                    if key not in variables.keys() or variables[key] is None:
                        raise RuntimeError(f"Task '{name}' did not set the value of '{key}'")
                return retval
            Task.registry[name] = wrapper
            return wrapper
        return real_decorator

    @staticmethod
    def register(category=TaskType.MISC):
        def real_decorator(function):
            name = function.__name__
            @wraps(function)
            def wrapper(*args, **kwargs):
                logger.info("Processing task: %s", function.__name__)
                retval = function(*args, **kwargs)
                # logger.debug("Processed task:", function.__name__)
                return retval
            Task.registry[name] = wrapper
            Task.types[name] = category
            return wrapper
        return real_decorator
