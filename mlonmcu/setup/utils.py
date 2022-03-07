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
import os
import signal
import sys
import multiprocessing
import subprocess

# import logging
import tarfile
import zipfile
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Union
from git import Repo

from mlonmcu import logging

logger = logging.get_logger()


def makeFlags(*args):
    """Resolve tuple-like arguments to a list of string.

    Parameters
    ----------
    args
        List of tuples of the form: [(True, "foo"), (False, "bar")]

    Returns
    -------
    dirname : str
        The generated directory name

    Examples
    --------
    >>> makeFlags((True, "foo"), (False, "bar"))
    ["foo"]
    """
    return [name for check, name in args if check]


def makeDirName(base: str, *args, flags: list = None) -> str:
    """Creates a directory name based on configuration values.

    Using snake_case style.

    Parameters
    ----------
    base : str
        Prefix of the filename to be generated.
    args
        List of tuples of the form: [(True, "foo"), (False, "bar")]
    flags : list
        Optional list of additional flags to be added.

    Returns
    -------
    dirname : str
        The generated directory name

    Examples
    --------
    >>> makeDirName("base", (True, "foo"), (False, "bar"), flags=["flag"])
    "base_foo_flag"
    """
    names = [base] + makeFlags(*args)
    if flags:
        names = names + flags
    return "_".join(names)


def exec(*args, **kwargs):
    """Execute a process with the given args and using the given kwards as Popen arguments.

    Parameters
    ----------
    args
        The command to be executed.
    """
    logger.debug("- Executing: " + str(args))
    subprocess.run([i for i in args], **kwargs, check=True)


def exec_getout(*args, live: bool = False, print_output: bool = True, handle_exit=None, **kwargs) -> str:
    """Execute a process with the given args and using the given kwards as Popen arguments and return the output.

    Parameters
    ----------
    args
        The command to be executed.
    live : bool
        If the stdout should be updated in real time.
    print_output : bool
        Print the output at the end on non-live mode.

    Returns
    -------
    output
        The text printed to the command line.
    """
    logger.debug("- Executing: " + str(args))
    outStr = ""
    if live:
        process = subprocess.Popen([i for i in args], **kwargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for line in process.stdout:
                new_line = line.decode(errors="replace")
                outStr = outStr + new_line
                print(new_line.replace("\n", ""))
            exit_code = None
            while exit_code is None:
                exit_code = process.poll()
            if handle_exit is not None:
                exit_code = handle_exit(exit_code)
            assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
                exit_code, " ".join(list(map(str, args)))
            )
        except KeyboardInterrupt as e:
            logger.debug("Interrupted subprocess. Sending SIGINT signal...")
            pid = process.pid
            os.kill(pid, signal.SIGINT)

    else:
        try:
            p = subprocess.Popen([i for i in args], **kwargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            outStr = p.communicate()[0].decode(errors="replace")
            exit_code = p.poll()
            # outStr = p.stdout.decode(errors="replace")
            if print_output:
                logger.debug(outStr)
            if handle_exit is not None:
                exit_code = handle_exit(exit_code)
            if exit_code != 0:
                logger.error(outStr)
            assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
                exit_code, " ".join(list(map(str, args)))
            )
        except KeyboardInterrupt as e:
            logger.debug("Interrupted subprocess. Sending SIGINT signal...")
            pid = p.pid
            os.kill(pid, signal.SIGINT)
        except subprocess.CalledProcessError as e:
            outStr = e.output.decode(errors="replace")
            logger.error(outStr)
            raise

    return outStr


def python(*args, **kwargs):
    """Run a python script with the current interpreter."""
    return exec_getout(sys.executable, *args, **kwargs)


# Makes sure all directories at the given path are created.
def mkdirs(path: Union[str, bytes, os.PathLike]):
    """Wrapper for os.makedirs which handels the special case where the path already exits."""
    if not os.path.exists(path):
        os.makedirs(path)


# Clones a git repository at given url into given dir and switches to the given branch.
def clone(
    url: str,
    dest: Union[str, bytes, os.PathLike],
    branch: str = "",
    recursive: bool = False,
    refresh: bool = False,
):
    """Helper function for cloning a repository.

    Parameters
    ----------
    url : str
        Clone URL of the repository.
    dest : Path
        Destination directory path.
    branch : str
        Optional branch name or commit reference/tag.
    recursive : bool
        If the clone should be done recursively.
    refesh : bool
        Enables switching the url/branch if the repo already exists
    """
    mkdirs(dest)

    if is_populated(dest):
        if refresh:
            repo = Repo(dest)
            # TODO: backup old remote?
            repo.remotes.origin.set_url(url)
            repo.remotes.origin.fetch()
            repo.git.checkout(branch)
            repo.git.pull("origin", branch)  # This should also work for specific commits
    else:
        if branch:
            repo = Repo.clone_from(url, dest, recursive=recursive, no_checkout=True)
            repo.git.checkout(branch)
            if recursive:
                output = repo.git.submodule("update", "--init", "--recursive")
        else:
            Repo.clone_from(url, dest, recursive=recursive)


def apply(
    repo_dir: Path,
    patch_file: Path,
):
    """Helper function for applying a patch to a repository.

    Parameters
    ----------
    repo_dir : Path
        Clone directory of repository.
    patch_file : Path
        Path to patch file.
    """

    repo = Repo(repo_dir)
    repo.git.clean("-xdf")  # Undo all changes
    repo.git.apply(patch_file)


def make(*args, threads=multiprocessing.cpu_count(), use_ninja=False, cwd=None, verbose=False, **kwargs):
    if cwd is None:
        raise RuntimeError("Please always pass a cwd to make()")
    if isinstance(cwd, Path):
        cwd = str(cwd.resolve())
    # TODO: make sure that ninja is installed?
    extraArgs = []
    tool = "ninja" if use_ninja else "make"
    extraArgs.append("-j" + str(threads))
    cmd = [tool] + extraArgs + list(args)
    exec_getout(*cmd, cwd=cwd, print_output=False, **kwargs)


def cmake(src, *args, debug=False, use_ninja=False, cwd=None, **kwargs):
    if cwd is None:
        raise RuntimeError("Please always pass a cwd to cmake()")
    if isinstance(cwd, Path):
        cwd = str(cwd.resolve())
    buildType = "Debug" if debug else "Release"
    extraArgs = []
    extraArgs.append("-DCMAKE_BUILD_TYPE=" + buildType)
    if use_ninja:
        extraArgs.append("-GNinja")
    cmd = ["cmake", str(src)] + extraArgs + list(args)
    exec_getout(*cmd, cwd=cwd, print_output=False, **kwargs)


# def move(a, b):  # TODO: make every utility compatible with Paths!
#     # This can not handle cross file-system renames!
#     if not isinstance(a, Path):
#         a = Path(a)
#     if not isinstance(b, Path):
#         b = Path(b)
#     a.replace(b)


def download(url, dest):
    urllib.request.urlretrieve(url, dest)


def extract(archive, dest):
    ext = Path(archive).suffix[1:]
    if ext == "zip":
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(dest)
    elif ext in ["tar", "gz", "xz", "tgz", "bz2"]:
        with tarfile.open(archive) as tar_file:
            tar_file.extractall(dest)
    else:
        raise RuntimeError("Unable to detect the archive type")


def remove(path):
    os.remove(path)


def move(src, dest):
    shutil.move(src, dest)


def copy(src, dest):
    shutil.copy(src, dest)


def is_populated(path):
    if not isinstance(path, Path):
        path = Path(path)
    return path.is_dir() and os.listdir(path.resolve())


def download_and_extract(url, archive, dest):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_archive = os.path.join(tmp_dir, archive)
        base_name = Path(archive).stem
        if ".tar" in base_name:
            base_name = Path(base_name).stem
        if url[-1] != "/":
            url += "/"
        download(url + archive, tmp_archive)
        extract(tmp_archive, tmp_dir)
        remove(os.path.join(tmp_dir, tmp_archive))
        mkdirs(dest.parent)
        if (Path(tmp_dir) / base_name).is_dir():  # Archive contains a subdirectory with the same name
            move(os.path.join(tmp_dir, base_name), dest)
        else:
            contents = list(Path(tmp_dir).glob("*"))
            if len(contents) == 1:
                tmp_dir_new = Path(tmp_dir) / contents[0]
                if tmp_dir_new.is_dir():  # Archive contains a single subdirectory with a different name
                    tmp_dir = tmp_dir_new
            move(tmp_dir, dest)
