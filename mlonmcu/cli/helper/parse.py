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
from mlonmcu.feature.features import (
    get_available_features,
)  # This does not really belong here
from mlonmcu.config import resolve_required_config


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
    else:
        return []


def extract_config(args):
    if args.config:
        configs = sum(args.config, [])
        configs = parse_vars(configs)
    else:
        configs = {}
    return configs


def extract_config_and_init_features(args, context=None):
    feature_names = extract_feature_names(args)
    config = extract_config(args)
    features = []
    for feature_name in feature_names:
        available_features = get_available_features(feature_name=feature_name)
        for feature_cls in available_features:
            required_keys = feature_cls.REQUIRED
            if len(required_keys) > 0:
                assert context is not None
                config.update(
                    resolve_required_config(
                        required_keys,
                        features=features,  # The order the features are provided is important here!
                        config=config,
                        cache=context.cache,
                    )
                )
            feature_inst = feature_cls(config=config)
            features.append(feature_inst)
    # How about FeatureType.other?

    return config, features
