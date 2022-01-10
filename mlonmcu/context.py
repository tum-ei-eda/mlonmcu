"""Definition if the contextmanager for mlonmcu environments."""

import os
import logging
import pathlib
import xdg
import filelock

from mlonmcu.session.run import Run
from mlonmcu.session.session import Session
from mlonmcu.setup.cache import TaskCache

#from mlonmcu.environment2 import load_environment_from_file
from mlonmcu.environment.loader import load_environment_from_file
from mlonmcu.environment.list import get_environments_map

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mlonmcu")
logger.setLevel(logging.DEBUG)

# TODO: sync across files
# config_dir = pathlib.Path(os.path.join(xdg.xdg_config_home(), "mlonmcu"))
# environments_dir = pathlib.Path(os.path.join(config_dir, "environments"))

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

    logger.debug("Looking for default environment for current user")
    envs_list = get_environments_map()
    if "default" in envs_list:
        assert "path" in envs_list["default"]
        directory = envs_list["default"]["path"]
        path = os.path.join(directory, "environment.yml")
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


def load_recent_sessions(env):
    sessions = []
    temp_directory = env.paths["temp"].path
    sessions_directory = temp_directory / "sessions"
    if not sessions_directory.is_dir():
        return []
    # TODO: in the future also strs (custom or hash) should be allowed
    session_ids = [int(o) for o in os.listdir(sessions_directory) if os.path.isdir(sessions_directory / o)]
    session_ids = sorted(session_ids)  # TODO: sort by session datetime?
    print("session_ids", session_ids)
    for sid in session_ids:
        session_directory = sessions_directory / str(sid)
        session_file = session_directory / "session.txt"
        if not session_file.is_file():
            continue
        runs_directory = session_directory / "runs"
        if runs_directory.is_dir():
            run_ids = [int(o) for o in os.listdir(runs_directory) if os.path.isdir(runs_directory / o)]
        else:
            run_ids = []
        print("run_ids", run_ids)
        runs = []
        for rid in run_ids:
            run_directory = runs_directory / str(rid)
            run_file = run_directory / "run.txt"
            run = Run()  # TODO
            runs.append(run)
        session = Session(idx=sid, archived=True, dir=session_directory)
        session.runs = runs
        sessions.append(session)
    return sessions

def resolve_environment_file(name=None, path=None):
    if name and path:
        raise RuntimeError("mlonmcu environments are specified either by name OR path")
    if name:
        env_file = get_environment_by_name(name)
    elif path:
        env_file = get_environment_by_path(path)
    else:
        env_file = lookup_environment()
        if not env_file:
            raise RuntimeError("Lookup for mlonmcu environment was not successful.")
    return env_file

class MlonMcuContext:
    """Contextmanager for mlonmcu environments."""
    def __init__(self, name=None, path=None, lock=False):
        """"Initialize MlonMcuContext."""
        env_file = resolve_environment_file(name=name, path=path)
        self.environment = load_environment_from_file(env_file)  # TODO: move to __enter__
        self.lock = lock
        self.lockfile = filelock.FileLock(os.path.join(self.environment.home, ".lock"))
        self.sessions = load_recent_sessions(self.environment)
        self.session_idx = self.sessions[-1].idx if len(self.sessions) > 0 else -1
        self.cache = TaskCache()

    def create_session(self):
        idx = self.session_idx + 1
        logger.debug("Creating a new session with idx %s", idx)
        temp_directory = self.environment.paths["temp"].path
        sessions_directory = temp_directory / "sessions"
        session_dir = sessions_directory / str(idx)
        session = Session(idx=idx, dir=session_dir)
        self.sessions.append(session)
        self.session_idx = idx
        # TODO: set latest symlink?

    def load_cache(self):
        if self.environment:
            if self.environment.paths:
                if "deps" in self.environment.paths:
                    deps_dir = self.environment.paths["deps"].path
                    if deps_dir.is_dir():
                        cache_file = deps_dir / "cache.ini"
                        if cache_file.is_file():
                            logger.info(f"Loading environment cache from file")
                            self.cache.read_from_file(cache_file)
                            logger.info(f"Successfully initialized cache")
                            return
        logger.info("No cache found in deps directory")

    def get_session(self):
        if self.session_idx < 0 or not self.sessions[-1].active:
            self.create_session()
        return self.sessions[-1]

    def __enter__(self):
        logger.debug("Enter MlonMcuContext")
        if self.lockfile.is_locked:
            raise RuntimeError(f"Current context is locked via: {self.lockfile.lock_file}")
        if self.lock:
            logger.debug("Locking context")
            try:
                self.lockfile.acquire(timeout=0)
            except filelock.Timeout as err:
                raise RuntimeError("Lock on current context could not be aquired.") from err
        self.load_cache()
        return self

    def cleanup(self):
        logger.debug("Cleaning up active sessions")
        for session in self.sessions:
            if session.active:
                session.close()

    @property
    def is_clean(self):
        return not any([sess.active for sess in self.sessions])


    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("Exit MlonMcuContext")
        self.cleanup()
        if self.lock:
            logger.debug("Releasing lock on context")
            self.lockfile.release()
        return False
