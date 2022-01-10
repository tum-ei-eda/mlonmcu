import os

# TODO: class TargetFactory:
from .common import execute

class Target:

    def __init__(self, name, features=[], config={}, context=None):
        self.name = name
        self.features = features
        self.config = config
        self.inspectProgram = "readelf"
        self.inspectprogramArgs = ["--all"]
        self.env = os.environ
        self.context = context

    def __repr__(self):
        return f"Target({self.name})"

    def exec(self, program, *args, **kwargs):
        raise NotImplementedError

    def inspect(self, program, *args, **kwargs):
        return execute(self.inspectProgram, program, *self.inspectprogramArgs, *args, **kwargs)
