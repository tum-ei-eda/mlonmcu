from functools import wraps
import itertools
from enum import Enum
import networkx as nx
import logging
from tqdm import tqdm
import time

logger = logging.getLogger('mlonmcu')
logger.setLevel(logging.DEBUG)


def get_combs(data):
    keys = list(data.keys())
    values = list(data.values())
    prod = list(itertools.product(*values))
    if len(prod) == 1:
        if len(prod[0]) == 0:
            prod = []
    combs = [dict(zip(keys, p)) for p in prod]
    return combs

class TaskType(Enum):
    MISC = 0
    FRAMEWORK = 1
    BACKEND = 2
    TOOLCHAIN = 3
    TARGET = 4
    FRONTEND = 5
    OPT = 6

class TaskGraph():

    def __init__(self, names, dependencies, providers):
        self.names = names
        self.dependencies = dependencies
        self.providers = providers

    def get_graph(self):
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

    def get_order(self):
       V, E = self.get_graph()
       G = nx.DiGraph(E)
       G.add_nodes_from(V)
       order = list(nx.topological_sort(G))
       return order



class TaskFactory():

    def __init__(self):
        self.registry = {}
        self.dependencies = {}
        self.providers = {}
        self.types = {}
        self.params = {}
        self.validates = {}
        self.changed = []  # Main problem: per

    def reset_changes(self):
        self.changed = []


    def needs(self, keys, force=True):
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
        return self.needs(keys, force=False)

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
        def real_decorator(function):
            name = function.__name__
            self.validates[name] = func
            return function
        return real_decorator

    def register(self, category=TaskType.MISC):
        def real_decorator(function):
            name = function.__name__
            @wraps(function)
            def wrapper(*args, progress=False, **kwargs):
                combs = get_combs(self.params[name])
                def get_valid_combs(combs):
                    ret = []
                    for comb in combs:
                        print("if", name, "in", self.validates)
                        if name in self.validates:
                            check = self.validates[name](args[0], params=comb)
                            if not check:
                                continue
                        ret.append(comb)
                    return ret
                combs = get_valid_combs(combs)
                def process(name_, params={}):
                    rebuild = False
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
                    from tqdm import tqdm
                    pbar = tqdm(total=max(len(combs), 1), desc='Processing', ncols=100, bar_format='{l_bar} {n_fmt}/{total_fmt}', leave=None)
                else:
                    pbar = None
                if len(combs) == 0:
                    if pbar:
                        pbar.set_description("Processing: %s" % name)
                    else:
                        logger.info("Processing task: %s", name)
                    time.sleep(0.1)
                    check = True
                    if name in self.validates:
                        check = self.validates[name](args[0], params={})
                    if check:
                        retval = process(name)
                    else:
                        retval = False
                    if pbar:
                        pbar.update(1)
                else:
                    for comb in combs:  # TODO process in parallel?
                        extended_name = name + str(comb)
                        if pbar:
                            pbar.set_description("Processing - %s" % extended_name)
                        else:
                            logger.info("Processing task: %s", extended_name)
                        time.sleep(0.1)
                        retval = process(extended_name, params=comb)
                        if pbar:
                            pbar.update(1)
                if pbar:
                    pbar.close()
                return retval
            self.registry[name] = wrapper
            self.types[name] = category
            self.params[name] = {}
            return wrapper
        return real_decorator
