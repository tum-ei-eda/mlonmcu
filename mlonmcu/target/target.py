"""MLonMCU Target definitions"""

import os
from pathlib import Path
from typing import List

from mlonmcu.context import MlonMcuContext
from mlonmcu.config import filter_config
from mlonmcu.feature.feature import FeatureType, Feature
from mlonmcu.feature.features import get_matching_features

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
        self.features = self.process_features(features)
        self.config = filter_config(
            self.config, self.name, self.DEFAULTS, self.REQUIRED
        )
        self.inspect_program = "readelf"
        self.inspect_program_args = ["--all"]
        self.env = os.environ
        self.context = context

    def __repr__(self):
        return f"Target({self.name})"

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.TARGET)
        for feature in features:
            assert (
                feature.name in self.FEATURES
            ), f"Incompatible feature: {feature.name}"
            feature.add_feature_config(self.name, self.config)
        return features

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
