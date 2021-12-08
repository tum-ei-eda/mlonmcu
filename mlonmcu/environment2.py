"""Environment module for mlonmcu."""

import os
import yaml
import pathlib

def load_environment_from_file(filename):
    """Utility to initialize a mlonmcu environment from a YAML file."""
    if isinstance(filename, str):
        filename = pathlib.Path(filename)
    with open(filename, encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
        if data:
            if "home" in data:
                print(data["home"], filename.parent)
                assert os.path.realpath(data["home"]) == os.path.realpath(filename.parent)
            else:
                data["home"] = filename.parent
            env = Environment(data)
            return env
        raise RuntimeError(f"Error opening environment file: {filename}")
    return None

class Environment:
    """Environment data structure for mlonmcu."""

    def __init__(self, data):
        if not data:
            raise RuntimeError("Invalid data")
        if "home" in data:
            self._home = data["home"]
        else:
            self._home = None
        self.paths = None
        self.repos = None
        self.features = None
        self.names = None

    @property
    def home(self):
        """Home directory of mlonmcu environment."""
        return self._home


    def to_yaml(self):
        """Convert mlonmcu environment to a YAML string."""
        raise NotImplementedError
