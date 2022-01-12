"""MLonMCU Target definitions"""

import os
from pathlib import Path
from typing import List

from mlonmcu.context import MlonMcuContext

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

    def __init__(
        self,
        name: str,
        features: List[str] = None,
        config: dict = None,
        context: MlonMcuContext = None,
    ):
        self.name = name
        self.features = features if features else []
        self.config = config if config else {}
        self.inspect_program = "readelf"
        self.inspect_program_args = ["--all"]
        self.env = os.environ
        self.context = context

    def __repr__(self):
        return f"Target({self.name})"

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
