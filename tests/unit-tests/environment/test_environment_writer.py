import os
import logging

from mlonmcu.environment.environment import Environment
from mlonmcu.environment.config import DefaultsConfig, PathConfig, RepoConfig
from mlonmcu.environment.writer import create_environment_dict


class MyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._home = "/foo/bar"
        self.defaults = DefaultsConfig(
            log_level=logging.DEBUG,
            log_to_file=False,
            log_rotate=False,
            default_framework=None,
            default_backends={},
            default_target=None,
            cleanup_auto=False,
            cleanup_keep=100,
        )
        self.paths = {"foo": PathConfig("bar"), "foobar": [PathConfig("baz"), PathConfig("baz2")]}
        self.repos = {"repo1": RepoConfig("repo1url"), "repo2": RepoConfig("repo2url", ref="repo2ref")}
        self.frameworks = []
        self.frontends = []
        self.toolchains = {}
        self.platforms = []
        self.targets = []
        self.vars = {"key": "val"}
        self.flags = {"my.var": ["foo", "bar"]}


def test_create_environment_dict():
    env = MyEnvironment()
    assert create_environment_dict(env) == {
        "home": "/foo/bar",
        "logging": {"level": "DEBUG", "to_file": False, "rotate": False},
        "cleanup": {"auto": False, "keep": 100},
        "paths": {
            "foo": os.path.join(os.getcwd(), "bar"),
            "foobar": [
                os.path.join(os.getcwd(), "baz"),
                os.path.join(os.getcwd(), "baz2"),
            ],
        },
        "repos": {
            "repo1": {"url": "repo1url", "ref": None, "options": {}},
            "repo2": {"url": "repo2url", "ref": "repo2ref", "options": {}},
        },
        "frameworks": {"default": None},
        "frontends": {},
        "toolchains": {},
        "platforms": {},
        "targets": {"default": None},
        "vars": {"key": "val"},
        "flags": {"my.var": ["foo", "bar"]},
    }
