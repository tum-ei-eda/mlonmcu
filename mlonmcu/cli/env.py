"""Command line subcommand for manaing environments."""

import os
import configparser
import xdg

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
