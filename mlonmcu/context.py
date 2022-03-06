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
"""Definition if the contextmanager for mlonmcu environments."""

import sys
import os
import shutil
import tempfile
from typing import List, Union
from pathlib import Path
import filelock

from mlonmcu.utils import ask_user
from mlonmcu.logging import get_logger, set_log_file
from mlonmcu.session.run import Run
from mlonmcu.session.session import Session
from mlonmcu.setup.cache import TaskCache
import mlonmcu.setup.utils as utils

from mlonmcu.environment.environment import Environment, UserEnvironment

from mlonmcu.environment.list import get_environments_map
from mlonmcu.environment.config import get_environments_dir

logger = get_logger()


def lookup_environment() -> Environment:
    """Helper function to automatically find a suitable environment.

    This function is used if neither a name nor a path of the environment was specified by the user.
    The lookup follows a predefined order:
    - Check current working directory
    - Check MLONMCU_HOME environment variable
    - Default environment for current user

    Returns
    -------
    environment : Path
        The environment (if the lookup was successful).
    """
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


def get_environment_by_path(path: Union[str, Path]) -> Environment:
    """Utility to find an environment file using a supplied path.

    Parameters
    ----------
    path : str/Path
        The path of the environment (or its YAML file).

    Returns
    -------
    Environment:
        The environment (if the lookup was successful).
    """
    if isinstance(path, str):
        path = Path(path)
    assert isinstance(path, Path)
    if path.is_dir():
        path = path / "environment.yml"
    if path.is_file():
        return path
    return None


def get_environment_by_name(name: str) -> Environment:
    """Utility to find an environment file using a supplied name.

    Parameters
    ----------
    name : str
        The name/alias if the environment.

    Returns
    -------
    Environment :
        The environment (if the lookup was successful).
    """
    # TODO: parse the ini file instead
    environments_dir = get_environments_dir()
    if environments_dir.is_dir():
        path = environments_dir / name
        if path.is_dir():
            return get_environment_by_path(path)
    return None


def get_ids(directory: Path) -> List[int]:
    """Get a sorted list of ids for sessions/runs found in the given directory.

    Parameters
    ----------
    directory : Path
        Directory where the sessions/runs are stored.

    Returns:
    list
        List of integers representing the session numbers. Empty list if directory does not exist.
    """
    if not directory.is_dir():
        return []

    ids = [int(o) for o in os.listdir(directory) if os.path.isdir(directory / o) and not os.path.islink(directory / o)]
    return sorted(ids)  # TODO: sort by session datetime?


def load_recent_sessions(env: Environment, count: int = None) -> List[Session]:
    """Get a list of recent sessions for the environment.

    Parameters
    ----------
    env : Environment
        MLonMCU environment which should be used.
    count : int
        Maximum number of sessions to return. Collect all if None.

    Returns
    -------
    list:
        The resulting list of session objects.
    """

    if count is not None:
        raise NotImplementedError()
    sessions = []

    sessions_directory = env.paths["temp"].path / "sessions"

    # TODO: in the future also strs (custom or hash) should be allowed
    session_ids = get_ids(sessions_directory)

    for sid in session_ids:
        session_directory = sessions_directory / str(sid)
        # session_file = sessions_directory / str(sid) / "session.txt"
        # if not session_file.is_file():
        #     continue
        runs_directory = session_directory / "runs"
        run_ids = get_ids(runs_directory)
        runs = []
        for rid in run_ids:
            run_directory = runs_directory / str(rid)
            run_file = run_directory / "run.txt"
            # run = Run.from_file(run_file)  # TODO: actually implement run restore
            run = Run()  # TODO: fix
            run.archived = True
            run.dir = run_directory
            runs.append(run)
        session = Session(idx=sid, archived=True, dir=session_directory)
        session.runs = runs
        session.dir = session_directory
        sessions.append(session)
    return sessions


def resolve_environment_file(name: str = None, path: str = None) -> Path:
    """Utility to find the environment file by a optionally given name or path.

    The lookup is performed in a predefined order:
    - If specified: name/path
    - Else: see lookup_environment()

    Parameters
    ----------
    name : str
        Hint for the environment name provided by the user.
    path : str
        Hint for the environment path provided by the user.

    Returns
    -------
    Path :
        Path to the found environment.yml (if sucessful)
    """
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


def setup_logging(environment):
    """Check logging settings for environment and initialize the logs directory.

    Attributes
    ----------
    environment : Environment
        The MLonMCU Environment where paths, repos, features,... are configured.
    """
    defaults = environment.defaults
    level = defaults.log_level
    to_file = defaults.log_to_file
    rotate = defaults.log_rotate
    if to_file:
        assert "logs" in environment.paths, "To use a logfile, define a logging directory in your environment.yml"
        directory = environment.paths["logs"].path
        if not directory.is_dir():
            directory.mkdir()
        path = directory / "mlonmcu.log"
        set_log_file(path, level=level, rotate=rotate)


class MlonMcuContext:
    """Contextmanager for mlonmcu environments.

    Attributes
    ----------
    environment : Environment
        The MLonMCU Environment where paths, repos, features,... are configured.
    lock : bool
        Holds if the environment should be limited to only one user or not.
    lockfile : FileLock
        The lock for the environment directory (optional).
    sessions : list
        List of sessions for the current environment.
    session_idx : list
        A counter for determining the next session index.
    cache : TaskCache
        The cache where paths of installed dependencies can be looked up.


    """

    def __init__(self, name: str = None, path: str = None, lock: bool = False):
        env_file = resolve_environment_file(name=name, path=path)
        assert env_file is not None, "Unable to find a MLonMCU environment"
        self.environment = UserEnvironment.from_file(env_file)  # TODO: move to __enter__
        setup_logging(self.environment)
        self.lock = lock
        self.lockfile = filelock.FileLock(os.path.join(self.environment.home, ".lock"))
        self.sessions = load_recent_sessions(self.environment)
        self.session_idx = self.sessions[-1].idx if len(self.sessions) > 0 else -1
        logger.debug(f"Restored {len(self.sessions)} recent sessions")
        self.cache = TaskCache()

    def create_session(self):
        """Create a new session in the current context."""
        idx = self.session_idx + 1
        logger.debug("Creating a new session with idx %s", idx)
        temp_directory = self.environment.paths["temp"].path
        sessions_directory = temp_directory / "sessions"
        session_dir = sessions_directory / str(idx)
        session = Session(idx=idx, dir=session_dir)
        self.sessions.append(session)
        self.session_idx = idx
        # TODO: move this to a helper function
        session_link = sessions_directory / "latest"
        if os.path.islink(session_link):
            os.unlink(session_link)
        os.symlink(session_dir, session_link)
        return session

    def load_cache(self):
        """If available load the cache.ini file in the deps directory"""
        if self.environment:
            if self.environment.paths:
                if "deps" in self.environment.paths:
                    deps_dir = self.environment.paths["deps"].path
                    if deps_dir.is_dir():
                        cache_file = deps_dir / "cache.ini"
                        if cache_file.is_file():
                            logger.info("Loading environment cache from file")
                            self.cache.read_from_file(cache_file)
                            logger.info("Successfully initialized cache")
                            return
        logger.info("No cache found in deps directory")

    def get_session(self, resume=False) -> Session:
        """Get an active session if available, else create a new one.

        Returns
        -------
        Session:
            An active session
        """
        if resume:
            assert len(self.sessions) > 0, "There is no recent session available"
            session = self.sessions[-1]
            assert False, "The latest session can not be resumed"
            raise NotImplementedError

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
        """Clean up the context before leaving the context by closing all active sessions"""
        logger.debug("Cleaning up active sessions")
        for session in self.sessions:
            if session.active:
                session.close()

    @property
    def is_clean(self):
        """Return true if all sessions in the context are inactive"""
        return not any(sess.active for sess in self.sessions)

    # WARNING: this will remove the actual session directories!
    def cleanup_sessions(self, keep=10, interactive=True):
        """Utility to cleanup old sessions from the disk."""
        assert self.is_clean
        all_sessions = self.sessions
        # print("all_sessions", all_sessions)
        to_keep = all_sessions[-keep:] if keep > 0 else []
        to_remove = self.sessions[:-keep] if keep > 0 else self.sessions
        count = len(to_remove)
        if count > 0:
            temp_dir = self.environment.lookup_path("temp").path
            sessions_dir = temp_dir / "sessions"
            print(f"The following {count} sessions will be removed from the environments temp directory ({temp_dir}):")
            print(" ".join([str(session.idx) for session in to_remove]))

            if ask_user("Are your sure?", default=not interactive, interactive=interactive):
                for session in to_remove:
                    session_dir = sessions_dir / str(session.idx)
                    if not session_dir.is_dir():
                        # Skip / Dir does not exist
                        continue
                    shutil.rmtree(session_dir)
                self.sessions = to_keep
                self.session_idx = self.sessions[-1].idx if len(self.sessions) > 0 else -1
                print("Done")
            else:
                print("Aborted")
        else:
            print("No sessions selected for removal")
        # We currently do not support rewirting the indices to start from scratch again as this would lead to inconsitencies with the path in the report/cmake build dirtectory

    def export(self, dest, session_ids=None, run_ids=None, interactive=True):
        dest = Path(dest)
        if (dest.is_file() and dest.exists()) or (dest.is_dir() and utils.is_populated(dest)):
            if not ask_user("Destination is already populated! Overwrite?", default=True, interactive=interactive):
                print("Aborted")
                return
        dest_, ext = os.path.splitext(dest)
        if session_ids is None:
            # Can not select all sessions, fall back to latest session
            session_ids = [-1]

        if run_ids is not None:
            assert len(session_ids) == 1, "Can only choose runs of a single session"

        def find_session(sid):
            if len(self.sessions) == 0:
                return None

            if sid == -1:
                assert len(self.sessions) > 0
                return self.sessions[-1]

            for session in self.sessions:
                if session.idx == sid:
                    return session
            return None

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            for sid in session_ids:
                session = find_session(sid)
                if session is None:
                    print(
                        f"Lookup for session id {sid} failed. Available:", " ".join([str(s.idx) for s in self.sessions])
                    )
                    sys.exit(1)
                if len(session_ids) == 1:
                    base = tmpdir
                else:
                    base = tmpdir / str(sid)
                if run_ids is None:
                    src = session.dir / "runs"
                    shutil.copytree(
                        src, base, dirs_exist_ok=True, symlinks=True
                    )  # Warning: dirs_exist_ok=True requires python 3.8+
                else:
                    base = base / "runs"
                    for rid in run_ids:
                        if rid >= len(session.runs):
                            print(
                                f"Lookup for run id {rid} failed in session {sid}. Available:",
                                " ".join([str(i) for i in range(len(session.runs))]),
                            )
                            sys.exit(1)
                        run = session.runs[rid]  # TODO: We currently do not check if the index actually exists
                        if len(run_ids) == 1 and len(session_ids) == 1:
                            run_base = tmpdir
                        else:
                            run_base = base / str(rid)
                        src = run.dir
                        shutil.copytree(
                            src, run_base, dirs_exist_ok=True
                        )  # Warning: dirs_exist_ok=True requires python 3.8+
            if ext in [".zip", ".tar"]:
                print(f"Creating archive: {dest}")
                shutil.make_archive(dest_, ext[1:], tmpdirname)
            else:
                print(f"Creating directory: {dest}")
                if dest.is_dir():
                    shutil.rmtree(dest)  # Cleanup old contents
                # dest.mkdir(exist_ok=True)
                shutil.move(tmpdirname, str(dest))
        print("Done")

    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("Exit MlonMcuContext")
        self.cleanup()
        if self.lock:
            logger.debug("Releasing lock on context")
            self.lockfile.release()
        return False
