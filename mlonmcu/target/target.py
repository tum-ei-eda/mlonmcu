"""MLonMCU Target definitions"""

import os
from pathlib import Path
from typing import List

from mlonmcu.context import MlonMcuContext
from mlonmcu.feature.feature import FeatureType, Feature

# TODO: class TargetFactory:
from .common import execute


class Target:
    """Base target class

    Attributes
    ----------
    name : str
        Default name of the target
    features : list
        List of target features which should be enabled
    config : dict
        User config defined via key-value pairs
    inspect_program : str
        Program which can be used to inspect executables (i.e. readelf)
    inspect_program_args : list
        List of additional arguments to the inspect_program
    env : os._Environ
        Optinal map of environment variables
    context : MlonMcuContext
        Optional context for resolving dependency paths
    """

    FEATURES = []
    DEFAULTS = {}
    REQUIRED = []

    def __init__(
        self,
        name: str,
        features: List[Feature] = None,
        config: dict = None,
        context: MlonMcuContext = None,
    ):
        self.name = name
        self.config = config if config else {}
        self.features = features if features else []
        self.process_features()
        self.filter_config()
        self.inspect_program = "readelf"
        self.inspect_program_args = ["--all"]
        self.env = os.environ
        self.context = context

    def __repr__(self):
        return f"Target({self.name})"

    def process_features(self):
        self.features = [
            feature
            for feature in self.features
            if FeatureType.BACKEND in feature.types()
        ]
        for feature in self.features:
            feature.add_feature_config(self.name, self.config)

    def remove_config_prefix(self, config):
        def helper(key):
            return key.split(f"{self.name}.")[-1]

        return {
            helper(key): value
            for key, value in config.items()
            if f"{self.name}." in key
        }

    def filter_config(self):
        cfg = self.remove_config_prefix(self.config)
        for required in self.REQUIRED:
            value = None
            if required in cfg:
                value = cfg[required]
            elif required in self.config:
                value = self.config[required]
                cfg[required] = value
            assert value is not None, f"Required config key can not be None: {required}"

        for key in self.DEFAULTS:
            if key not in cfg:
                cfg[key] = self.DEFAULTS[key]

        for key in cfg:
            if key not in list(self.DEFAULTS.keys()) + self.REQUIRED:
                logger.warn("Target received an unknown config key: %s", key)
                del cfg[key]

        self.config = cfg

    def exec(self, program: Path, *args, **kwargs):
        """Use target to execute a executable with given arguments"""
        raise NotImplementedError

    def inspect(self, program: Path, *args, **kwargs):
        """Use target to inspect a executable"""
        return execute(
            self.inspect_program, program, *self.inspect_program_args, *args, **kwargs
        )

    def get_cmake_args(self):
        return [f"-DTARGET_SYSTEM={self.name}"]
