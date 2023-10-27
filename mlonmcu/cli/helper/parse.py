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
import itertools
from collections import ChainMap

NUM_GEN_ARGS = 9


def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split("=")
    key = items[0].strip()  # we remove blanks around keys, as is logical
    if len(key) == 0 or len(items) <= 1 or "=" not in s:
        raise RuntimeError(f"The argument {s} is not a key-value pair")
    if len(items) > 1:
        # rejoin the rest:
        value = "=".join(items[1:])
    return (key, value)


def parse_vars(items):  # TODO: this needs to be used in other subcommands as well?
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            if len(item) > 0:
                key, value = parse_var(item)
                d[key] = value
    return d


def extract_feature_names(args):
    if args.feature:
        features = args.feature
    else:
        features = []

    def helper(args, name):
        gen = []
        if hasattr(args, name):
            feature_gen = getattr(args, name)
            if not feature_gen:
                gen = [[]]
                return gen
            for x in feature_gen:
                if "_" in x:
                    assert len(set(x)) == 1
                    gen.append([])
                else:
                    gen.append(x)
        else:
            gen = [[]]
        return gen

    gens = []
    for i in range(NUM_GEN_ARGS):
        suffix = str(i + 1) if i > 0 else ""
        gen = helper(args, "feature_gen" + suffix)
        gens.append(gen)

    gen = list(map(lambda x: sum(x, []), (itertools.product(*gens))))

    return features, gen


def extract_config(args):
    if args.config:
        configs = sum(args.config, [])
        configs = parse_vars(configs)
    else:
        configs = {}

    def helper(args, name):
        gen = []
        if hasattr(args, name):
            config_gen = getattr(args, name)
            if not config_gen:
                gen = [{}]
                return gen
            for x in config_gen:
                if "_" in x:
                    assert len(set(x)) == 1
                    gen.append({})
                else:
                    c = parse_vars(x)
                    gen.append(c)
        else:
            gen = [{}]
        return gen

    gens = []
    for i in range(NUM_GEN_ARGS):
        suffix = str(i + 1) if i > 0 else ""
        gen = helper(args, "config_gen" + suffix)
        gens.append(gen)

    gen = list(map(lambda x: dict(ChainMap(*x)), (itertools.product(*gens))))

    return configs, gen


def extract_config_and_feature_names(args, context=None):
    # TODO: get features from context?
    feature_names, feature_gen = extract_feature_names(args)
    config, config_gen = extract_config(args)
    return config, feature_names, config_gen, feature_gen


def extract_frontend_names(args, context=None):
    frontend_names = args.frontend
    names = []
    if isinstance(frontend_names, list) and len(frontend_names) > 0:
        names = frontend_names
    elif isinstance(frontend_names, str):
        names = [frontend_names]
    else:
        # No need to specify a default, because we just use the provided order in the environment.yml
        assert frontend_names is None, "TODO"
        assert context is not None, "Need context to resolve default frontends"
        all_frontend_names = context.environment.lookup_frontend_configs(names_only=True)
        names.extend(all_frontend_names)
    return names


def extract_postprocess_names(args, context=None):
    return list(dict.fromkeys(args.postprocess)) if args.postprocess is not None else []


def extract_backend_names(args, context=None):
    if isinstance(args.backend, list) and len(args.backend) > 0:
        backends = args.backend
    elif isinstance(args.backend, str):
        backends = [args.backend]
    else:
        assert args.backend is None, "TODO"
        assert context is not None
        frameworks = context.environment.get_default_frameworks()
        backends = []
        for framework in frameworks:
            framework_backends = context.environment.get_default_backends(framework)
            backends.extend(framework_backends)
    return backends


def extract_target_names(args, context=None):
    if isinstance(args.target, list) and len(args.target) > 0:
        targets = args.target
    elif isinstance(args.target, str):
        targets = [args.target]
    else:
        assert args.target is None, "TODO"
        # assert context is not None
        if context is None:
            return [None]
        targets = context.environment.get_default_targets()
    return targets


def extract_platform_names(args, context=None):
    if args.platform:
        platforms = [x[0] for x in args.platform]
    else:
        assert args.platform is None
        if context is None:
            return [None]
        platforms = context.environment.lookup_platform_configs(names_only=True)
    return platforms
