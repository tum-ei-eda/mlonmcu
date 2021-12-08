"""Definition if the contextmanager for mlonmcu environments."""

import os
import logging
import pathlib
import xdg
import filelock

#from mlonmcu.environment2 import load_environment_from_file
from mlonmcu.environment.loader import load_environment_from_file

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mlonmcu")
logger.setLevel(logging.DEBUG)

# TODO: sync across files
config_dir = pathlib.Path(os.path.join(xdg.XDG_CONFIG_HOME, "mlonmcu"))
environments_dir = pathlib.Path(os.path.join(config_dir, "environments"))

def lookup_environment():
    """Helper function to automatically find a suitable environment."""
    logger.debug("Starting lookup for mlonmcu environment")

    logger.debug("First checking in local working directory")
    path = os.path.join(os.getcwd(), "environment.yml")
    if os.path.exists(path):
        logger.debug("Found environment directory: %s", path)
        return path

    logger.debug("Next checking environment variables")
    home = os.environ.get("MLONMCU_HOME")
    if home:
        path = os.path.join(home, "environment.yml")
        if os.path.exists(path):
            logger.debug("Found environment directory: %s", path)
            return path
    return None


def get_environment_by_path(path):
    """Utility to find an environment file using a supplied path."""
    if isinstance(path, str):
        path = pathlib.Path(path)
    assert isinstance(path, pathlib.Path)
    if path.is_dir():
        path = path / "environment.yml"
    if path.is_file():
        return path
    return None


def get_environment_by_name(name):
    """Utility to find an environment file using a supplied name."""
    if environments_dir.is_dir():
        path = environments_dir / name
        if path.is_dir():
            return get_environment_by_path(path)
    return None


class MlonMcuContext:
    """Contextmanager for mlonmcu environments."""
    def __init__(self, name=None, path=None, lock=False):
        """"Initialize MlonMcuContext."""
        if name and path:
            raise RuntimeError("mlonmcu environments are specified either by name OR path")
        if name:
            env_file = get_environment_by_name(name)
        elif path:
            env_file = get_environment_by_name(path)
        else:
            env_file = lookup_environment()
            if not env_file:
                raise RuntimeError("Lookup for mlonmcu environment was not successfull.")

        self.environment = load_environment_from_file(env_file)
        self.lock = lock
        self.lockfile = filelock.FileLock(os.path.join(self.environment.home, ".lock"))

    def __enter__(self):
        logger.debug("Enter MlonMcuContext")
        if self.lockfile.is_locked:
            raise RuntimeError(f"Current context is locked via: {self.lockfile.lock_file}")
        if self.lock:
            logger.debug("Locking context")
            try:
                self.lockfile.acquire(timeout=0)
            except filelock.Timeout as err:
                raise RuntimeError("Lock on current conext could not be aquired.") from err
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("Exit MlonMcuContext")
        if self.lock:
            logger.debug("Releasing lock on context")
            self.lockfile.release()
        return False
