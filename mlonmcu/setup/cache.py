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
"""Definition of Taks Cache"""

import os
import configparser
from typing import Any


def convert_key(name):
    if not isinstance(name, tuple):
        name = (name, frozenset())
    else:
        assert len(name) == 2
        if not isinstance(name[1], frozenset):
            name = (name[0], frozenset(name[1]))
    return name


class TaskCache:
    """Task cache used to store dependency paths for the current and furture sessions.

    This can be interpreted as a "modded" dictionary which takes a key + some flags.
    """

    def __init__(self):
        self._vars = {}

    def __repr__(self):
        return str(self._vars)

    def __setitem__(self, name, value):
        name = convert_key(name)
        self._vars[name[0]] = value  # Holds latest value
        self._vars[name] = value

    def __delitem__(self, name):
        name = convert_key(name)
        del self._vars[name]

    def __getitem__(self, name):
        name = convert_key(name)
        return self._vars[name]

    def __len__(self):
        return len(self._vars)

    def __contains__(self, name):
        name = convert_key(name)
        return name in self._vars.keys()

    def find_best_match(self, name: str, flags=[]) -> Any:
        """Utility whih tries to resolve the cache entry with the beste match.

        Parameters
        ----------
        name : str
            The cache-key.
        flags : list
            Optional flags used for the lookup.
        """
        # print("find_best_match", name, flags)
        keys = self._vars.keys()
        # print("keys", keys)
        matches = []
        counts = []
        for key in keys:
            if not isinstance(key, tuple):
                continue
            assert len(key) == 2
            name_, flags_ = key[0], key[1]
            if name == name_:
                count = 0
                for flag in flags_:
                    if flag not in flags:
                        count = -1  # incompatible
                        break
                    count = count + 1
                if count >= 0:
                    matches.append(flags_)
                    counts.append(count)
        if len(counts) == 0:
            raise RuntimeError("Unable to find a match in the cache")
        m = max(counts)
        assert counts.count(m) == 1, f"For the given set of flags, there are multiple cache matches for the name {name}"
        idx = counts.index(m)
        flag = matches[idx]
        ret = self._vars[name, flag]
        return ret

    def read_from_file(self, filename, reset=True):
        if reset:
            self._vars = {}
        if not os.path.isfile(filename):
            raise RuntimeError(f"File not found: {filename}")
        cfg = configparser.ConfigParser()
        cfg.read(filename)
        sections = cfg.sections()
        for section in sections:
            if section == "default":
                flags = set()
            else:
                flags = {flag for flag in section.split(",")}
            content = dict(cfg[section].items())
            for name, value in content.items():
                self[name, flags] = value

    def write_to_file(self, filename):
        # d = self._vars

        out = {}  # This will be a dict of dicts
        for key in self._vars:
            # print(key, type(key))
            if isinstance(key, str):
                continue
            name, flags = key[0], key[1]
            value = self._vars[key]
            if len(flags) == 0:
                section_name = "default"
            else:
                section_name = ",".join(sorted(flags))
            if section_name in out:
                out[section_name][name] = value
            else:
                out[section_name] = {name: value}

        with open(filename, "w") as cachefile:
            cfg = configparser.ConfigParser()
            if "default" in out:  # Default section should be first
                cfg["default"] = out["default"]
            for x in out:
                if x == "default":
                    continue
                cfg[x] = out[x]
            cfg.write(cachefile)
