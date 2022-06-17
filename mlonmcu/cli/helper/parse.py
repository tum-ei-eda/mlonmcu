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


def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    assert "=" in s, "Not a key-value pair"
    items = s.split("=")
    key = items[0].strip()  # we remove blanks around keys, as is logical
    assert len(key) > 0, "Empty key"
    assert len(items) > 1, "Not enough items"
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
        return args.feature
    return []


def extract_config(args):
    if args.config:
        configs = sum(args.config, [])
        configs = parse_vars(configs)
    else:
        configs = {}
    return configs


def extract_config_and_feature_names(args, context=None):
    # TODO: get features from context?
    feature_names = extract_feature_names(args)
    config = extract_config(args)
    return config, feature_names


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
    return list(set(args.postprocess)) if args.postprocess is not None else []


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
