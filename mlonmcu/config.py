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
"""Collection of utilities to manage MLonMCU configs."""
import distutils.util
import ast

from mlonmcu.feature.type import FeatureType
from mlonmcu.logging import get_logger

logger = get_logger()


def remove_config_prefix(config, prefix, skip=None):
    """Iterate over keys in dict and remove given prefix.

    Arguments
    ---------
    config : dict
        The configuration data.
    prefix : str
        The prefix to remove.
    skip : List[str], optional
        A list of keys which should not be altered.

    Returns
    -------
    ret : dict
        The transformed configuration.

    """
    if skip is None:
        skip = []

    def helper(key):
        return key.split(f"{prefix}.")[-1]

    return {helper(key): value for key, value in config.items() if f"{prefix}." in key and key not in skip}


def filter_config(config, prefix, defaults, optionals, required_keys):
    """Filter the global config for a given component prefix.

    Arguments
    ---------
    config : dict
        The configuration data.
    prefix : str
        The prefix for the component.
    defaults : dict
        The default values used if not overwritten by user.
    optionals : dict
        The optional keys for the component.
    required_keys : list
        The required keys for the component.

    Returns
    -------
    cfg : dict
        The filteres configuration.

    Raises
    ------
    AssertionError: If a required key is missing.
    """
    if not isinstance(required_keys, set):
        logger.warning(
            "Deprecated: FEATURES, REQUIRED, OPTIONAL should now be sets, not lists for component %s", prefix
        )
        required_keys = set(required_keys)
    if not isinstance(optionals, set):
        optionals = set(optionals)
    cfg = remove_config_prefix(config, prefix, skip=required_keys | optionals)
    for required in required_keys:
        value = None
        if required in cfg:
            value = cfg[required]
        elif required in config:
            value = config[required]
            cfg[required] = value
        assert value is not None, f"Required config key can not be None: {required}"

    for optional in optionals:
        if optional in cfg:
            continue
        if optional in config:
            value = config[optional]
            cfg[optional] = value
        else:
            cfg[optional] = None

    for key in defaults:
        if key not in cfg:
            cfg[key] = defaults[key]

    for key in cfg:
        if key not in set(defaults.keys()) | required_keys:
            # logger.warn("Component received an unknown config key: %s", key)
            pass

    return cfg


def resolve_required_config(
    required_keys, optional=None, features=None, config=None, cache=None, hints=None, default_flags=None
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

    hints : List[str]
        List of additional flags which can be provided as a hint to lookup a cache config.

    default_flags : dict
        User-provided mapping of cache flags for some cache entries.

    Returns
    -------
    result : dict

    """

    def get_sublists(in_list):
        ret = [[]]
        for i in range(len(in_list) + 1):
            for j in range(i + 1, len(in_list) + 1):
                sub = in_list[i:j]
                ret.append(sub)
        return ret

    hint_combinations = get_sublists(hints if hints else [])

    def get_cache_flags(features, default):
        result = {}
        if default:
            # result.update({key: frozenset(value) for key, value in default.items()})
            result.update(default)
        if features:
            for feature in features:
                if FeatureType.SETUP in type(feature).types():
                    feature.add_required_cache_flags(result)
        return result

    ret = {}
    cache_flags = get_cache_flags(features, default_flags)
    for key in required_keys:
        if config is None or key not in config:
            assert cache is not None, "No dependency cache was provided. Either provide a cache or config."
            if len(cache) == 0:
                raise RuntimeError("The dependency cache is empty! Make sure `to run `mlonmcu` setup first.`")
            flags = cache_flags.get(key, ())
            value = None
            for hint_combination in hint_combinations:
                if (key, tuple(list(flags) + hint_combination)) in cache:
                    value = cache[key, flags]
                    break
            if value is None:
                if len(flags) == 0:
                    raise RuntimeError(
                        f"Dependency cache miss for required key '{key}'. Try re-running `mlonmcu setup`."
                    )
                else:
                    raise RuntimeError(
                        f"Dependency cache miss for required key '{key}' with flags {flags}."
                        + " Try re-running `mlonmcu setup`."
                    )
            ret[key] = value
        else:
            ret[key] = config[key]
    if optional:
        for key in optional:
            if config is None or key not in config:
                assert cache is not None, "No dependency cache was provided. Either provide a cache or config."
                if cache is None or len(cache) == 0:
                    continue
                flags = cache_flags.get(key, ())
                value = None
                for hint_combination in hint_combinations:
                    if (key, tuple(list(flags) + hint_combination)) in cache:
                        value = cache[key, flags]
                        break
                if value is None:
                    continue
                ret[key] = value
            else:
                ret[key] = config[key]

    return ret


def str2bool(value, allow_none=False):
    if value is None:
        assert allow_none, "str2bool received None value while allow_none=False"
        return value
    if isinstance(value, (int, bool)):
        return bool(value)
    assert isinstance(value, str)
    return bool(distutils.util.strtobool(value))


def str2dict(value, allow_none=False):
    if value is None:
        assert allow_none, "str2dict received None value while allow_none=False"
        return value
    if isinstance(value, dict):
        return value
    assert isinstance(value, str)
    # TODO: parse key=value,key2=value2 via regex
    return dict(ast.literal_eval(value))


def str2list(value, allow_none=False):
    if value is None:
        assert allow_none, "str2list received None value while allow_none=False"
        return value
    if isinstance(value, (list, set, tuple)):
        return list(value)
    assert isinstance(value, str)
    if value.startswith("["):
        return list(ast.literal_eval(value))
    return value.split(",")
