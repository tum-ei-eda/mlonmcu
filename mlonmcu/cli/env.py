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
"""Command line subcommand for manaing environments."""

import configparser

from mlonmcu.environment.config import get_environments_file


def get_parser(subparsers):
    """ "Define and return a subparser for the env subcommand."""
    parser = subparsers.add_parser("env", description="List ML on MCU environments.")
    parser.set_defaults(func=handle)
    return parser


class EnvironmentHint:
    def __init__(self, name, path, created_at=None):
        self.name = name
        self.path = path
        self.created_at = created_at


def lookup_user_environments(file):
    config = configparser.ConfigParser()
    config.read(file)
    hints = []
    for env_name in config.sections():
        env = config[env_name]
        assert "path" in env
        path = env["path"]
        created_at = None
        if "created" in env:
            created_at = env["created"]
        hint = EnvironmentHint(env_name, path, created_at=created_at)
        hints.append(hint)
    return hints


def handle(args):
    envs_file = get_environments_file()
    print(f"Looking for user environments config file: {envs_file}")
    envs = lookup_user_environments(envs_file)
    count = len(envs)
    if not envs or count == 0:
        print("No mlonmcu environments were found for the current user. You can create a new one using `mlonmcu init`")
    else:
        print(f"Found {count} mlonmcu environment(s):")
        for env in envs:
            print(f"    - {env.name} ({env.path})")

        print("Point the environment variable MLONMCU_HOME to a environment directory to get started")
