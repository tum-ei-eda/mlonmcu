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
from mlonmcu.feature.type import FeatureType
from mlonmcu.logging import get_logger

logger = get_logger()


def remove_config_prefix(config, prefix, skip=[]):
    def helper(key):
        return key.split(f"{prefix}.")[-1]

    return {helper(key): value for key, value in config.items() if f"{prefix}." in key and key not in skip}


def filter_config(config, prefix, defaults, required_keys):
    cfg = remove_config_prefix(config, prefix, skip=required_keys)
    for required in required_keys:
        value = None
        if required in cfg:
            value = cfg[required]
        elif required in config:
            value = config[required]
            cfg[required] = value
        assert value is not None, f"Required config key can not be None: {required}"

    for key in defaults:
        if key not in cfg:
            cfg[key] = defaults[key]

    for key in cfg:
        if key not in list(defaults.keys()) + required_keys:
            # logger.warn("Component received an unknown config key: %s", key)
            pass

    return cfg


def resolve_required_config(
    required_keys, features=None, targets=None, config=None, cache=None
):  # TODO: add framework, backend, and frontends as well?
    """Utility which iterates over a set of given config keys and
    resolves their values using the passed config and/or cache.

    Parameters
    ----------
    required_keys : List[str]

    features : List[Feature]

    config : dict

    cache : TaskCache
        Optional task cache parsed from the `cache.ini` file in the `deps` directory.

    Returns
    -------
    result : dict

    """

    def get_cache_flags(features, targets):
        result = {}
        if features:
            for feature in features:
                if FeatureType.SETUP in type(feature).types():
                    feature.add_required_cache_flags(result)
        return result

    ret = {}
    cache_flags = get_cache_flags(features, targets)
    for key in required_keys:
        if config is None or key not in config:
            assert cache is not None, "No dependency cache was provided. Either provide a cache or config."
            if len(cache) == 0:
                raise RuntimeError("The dependency cache is empty! Make sure `to run `mlonmcu` setup first.`")
            flags = cache_flags.get(key, ())
            if (key, flags) in cache:
                value = cache[key, flags]
                ret[key] = value
            else:
                raise RuntimeError(f"Dependency cache miss for required key '{key}'. Try re-running `mlonmcu setup`.")
        else:
            ret[key] = config[key]

    return ret
