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
from typing import Union, List, Callable, Optional
from git import Repo
from tqdm import tqdm

from mlonmcu import logging
from mlonmcu.environment.config import RepoConfig

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
    kwargs
        Parameters to be passed to subprocess
    """
    logger.warning("DEPRECATED: Please use utils.execute(..., ignore_output=True) instead of utils.exec(...)")
    # Original implementation
    # logger.debug("- Executing: " + str(args))
    # if "cwd" in kwargs:
    #     logger.debug("- CWD: " + str(kwargs["cwd"]))
    # subprocess.run([i for i in args], **kwargs, check=True)

    # Call new implementation
    _ = execute(*args, ignore_output=True, live=False, print_func=None, **kwargs)


def exec_getout(*args, live: bool = False, print_output: bool = False, handle_exit=None, prefix="", **kwargs) -> str:
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
    logger.warning("DEPRECATED: Please use utils.execute(...) instead of utils.exec_getout(...)")
    # Original implementation:
    # logger.debug("- Executing: " + str(args))
    # outStr = ""
    # if live:
    #     process = subprocess.Popen([i for i in args], **kwargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #     try:
    #         for line in process.stdout:
    #             new_line = prefix + line.decode(errors="replace")
    #             outStr = outStr + new_line
    #             print(new_line.replace("\n", ""))
    #         exit_code = None
    #         while exit_code is None:
    #             exit_code = process.poll()
    #         if handle_exit is not None:
    #             exit_code = handle_exit(exit_code)
    #         assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
    #             exit_code, " ".join(list(map(str, args)))
    #         )
    #     except KeyboardInterrupt:
    #         logger.debug("Interrupted subprocess. Sending SIGINT signal...")
    #         pid = process.pid
    #         os.kill(pid, signal.SIGINT)

    # else:
    #     try:
    #         p = subprocess.Popen([i for i in args], **kwargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #         outStr = p.communicate()[0].decode(errors="replace")
    #         exit_code = p.poll()
    #         # outStr = p.stdout.decode(errors="replace")
    #         if print_output:
    #             logger.debug(prefix + outStr)
    #         if handle_exit is not None:
    #             exit_code = handle_exit(exit_code)
    #         if exit_code != 0:
    #             logger.error(outStr)
    #         assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
    #             exit_code, " ".join(list(map(str, args)))
    #         )
    #     except KeyboardInterrupt:
    #         logger.debug("Interrupted subprocess. Sending SIGINT signal...")
    #         pid = p.pid
    #         os.kill(pid, signal.SIGINT)
    #     except subprocess.CalledProcessError as e:
    #         outStr = e.output.decode(errors="replace")
    #         logger.error(outStr)
    #         raise e
    # return outStr
    return execute(
        *args,
        ignore_output=False,
        live=live,
        # print_func=print,
        handle_exit=handle_exit,
        err_func=logger.error,
        prefix=prefix,
        **kwargs,
    )


def execute(
    *args: List[str],
    ignore_output: bool = False,
    live: bool = False,
    print_func: Callable = print,
    handle_exit: Optional[Callable] = None,
    err_func: Callable = logger.error,
    encoding: Optional[str] = "utf-8",
    stdin_data: Optional[bytes] = None,
    prefix: str = "",
    **kwargs,
) -> str:
    """Wrapper for running a program in a subprocess.

    Parameters
    ----------
    args : list
        The actual command.
    ignore_output : bool
        Do not get the stdout and stderr or the subprocess.
    live : bool
        Print the output line by line instead of only at the end.
    print_func : Callable
        Function which should be used to print sysout messages.
    handle_exit: Callable
        Handler for exit code.
    err_func : Callable
        Function which should be used to print errors.
    encoding: str, optional
        Used encoding for the stdout.
    stdin_data: bytes, optional
        Send this to the stdin of the process.
    kwargs: dict
        Arbitrary keyword arguments passed through to the subprocess.

    Returns
    -------
    out : str
        The command line output of the command
    """
    # TODO: catch keyboardinterrupt
    logger.debug("- Executing: %s", str(args))
    if "cwd" in kwargs:
        logger.debug("- CWD: %s", str(kwargs["cwd"]))
    # if "env" in kwargs:
    #     logger.debug("- ENV: %s", str(kwargs["env"]))
    if ignore_output:
        assert not live
        subprocess.run(args, **kwargs, check=True)
        return None

    def args_helper(x):
        x = str(x)
        if "[" in x or "]" in x or " " in x:
            x = f'"{x}"'
        return x

    out_str = ""
    if live:
        with subprocess.Popen(
            args,
            **kwargs,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            try:
                if stdin_data:
                    raise RuntimeError("stdin_data only supported if live=False")
                    # not working...
                    # process.stdin.write(stdin_data)
                for line in process.stdout:
                    if encoding:
                        line = line.decode(encoding, errors="replace")
                        new_line = prefix + line
                    else:
                        new_line = line
                    out_str = out_str + new_line
                    print_func(new_line.replace("\n", ""))
                exit_code = None
                while exit_code is None:
                    exit_code = process.poll()
                if handle_exit is not None:
                    out_str_ = out_str
                    if encoding is None:
                        out_str_ = out_str_.decode("utf-8", errors="ignore")
                    exit_code = handle_exit(exit_code, out=out_str_)
                assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
                    exit_code, " ".join(list(map(args_helper, args)))
                )
            except KeyboardInterrupt:
                logger.debug("Interrupted subprocess. Sending SIGINT signal...")
                pid = process.pid
                os.kill(pid, signal.SIGINT)
    else:
        try:
            p = subprocess.Popen(
                [i for i in args], **kwargs, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            if stdin_data:
                out_str = p.communicate(input=stdin_data)[0]
            else:
                out_str = p.communicate()[0]
            if encoding:
                out_str = out_str.decode(encoding, errors="replace")
                out_str = prefix + out_str
            exit_code = p.poll()
            # print_func(out_str)
            if handle_exit is not None:
                out_str_ = out_str
                if encoding is None:
                    out_str_ = out_str_.decode("utf-8", errors="ignore")
                exit_code = handle_exit(exit_code, out=out_str_)
            if exit_code != 0:
                err_func(out_str)
            assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
                exit_code, " ".join(list(map(args_helper, args)))
            )
        except KeyboardInterrupt:
            logger.debug("Interrupted subprocess. Sending SIGINT signal...")
            pid = p.pid
            os.kill(pid, signal.SIGINT)
        except subprocess.CalledProcessError as e:
            out_str = e.output.decode(errors="replace")
            err_func(out_str)
            raise e

    return out_str


def python(*args, **kwargs):
    """Run a python script with the current interpreter."""
    return execute(sys.executable, *args, **kwargs)


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
    submodules: list = [],
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
    submodules : list of strings
        Only affects when recursive is true. Submodules to be updated. If empty, all submodules will be updated.
    recursive : bool
        If the clone should be done recursively.
    refesh : bool
        Enables switching the url/branch if the repo already exists
    """
    mkdirs(dest)

    def update_submodules():
        if recursive:
            if submodules:
                for submodule in submodules:
                    assert isinstance(submodule, str), f"Submodules should be a list of str. {submodule} is not str."
                repo.git.submodule("update", "--init", "--recursive", "--", *submodules)
            else:
                repo.git.submodule("update", "--init", "--recursive")

    if is_populated(dest):
        if refresh:
            repo = Repo(dest)
            # TODO: backup old remote?
            repo.remotes.origin.set_url(url)
            repo.remotes.origin.fetch()
            repo.git.checkout(branch)
            repo.git.pull("origin", branch)  # This should also work for specific commits
            update_submodules()
    else:
        if branch:
            repo = Repo.clone_from(url, dest, recursive=recursive, no_checkout=True)
            repo.git.checkout(branch)
            update_submodules()
        else:
            Repo.clone_from(url, dest, recursive=recursive)


def clone_wrapper(cfg: RepoConfig, dest: Union[str, bytes, os.PathLike], refresh: bool = False):
    clone(cfg.url, dest, branch=cfg.ref, submodules=cfg.submodules, recursive=cfg.recursive, refresh=refresh)


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
    return execute(*cmd, cwd=cwd, **kwargs)


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
    return execute(*cmd, cwd=cwd, **kwargs)


# def move(a, b):  # TODO: make every utility compatible with Paths!
#     # This can not handle cross file-system renames!
#     if not isinstance(a, Path):
#         a = Path(a)
#     if not isinstance(b, Path):
#         b = Path(b)
#     a.replace(b)


def download(url, dest, progress=False):
    logger.debug("- Downloading: %s", url)

    def hook(t):
        """Wraps tqdm instance."""
        last_b = [0]

        def update_to(b=1, bsize=1, tsize=None):
            """
            b  : int, optional
                Number of blocks transferred so far [default: 1].
            bsize  : int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize  : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
            """
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return update_to

    if progress:
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading File") as t:
            urllib.request.urlretrieve(url, dest, reporthook=hook(t))
    else:
        urllib.request.urlretrieve(url, dest)


def extract(archive, dest, progress=False):
    ext = Path(archive).suffix[1:]

    def handle(f):
        if progress:
            members = f.getmembers()
            for m in tqdm(iterable=members, total=len(members), desc="Extracting..."):
                f.extract(m, dest)
        else:
            f.extractall(dest)

    if ext == "zip":
        with zipfile.ZipFile(archive) as zip_file:
            handle(zip_file)
    elif ext in ["tar", "gz", "xz", "tgz", "bz2"]:
        with tarfile.open(archive) as tar_file:
            handle(tar_file)
    else:
        raise RuntimeError("Unable to detect the archive type")


def remove(path):
    os.remove(path)


def move(src, dest):
    shutil.move(src, dest)


def copy(src, dest):
    shutil.copy(src, dest)


def symlink(src, dest):
    os.symlink(src, dest)


def is_populated(path):
    if not isinstance(path, Path):
        path = Path(path)
    return path.is_dir() and os.listdir(path.resolve())


def download_and_extract(url, archive, dest, progress=False, force=True):
    if isinstance(dest, str):
        dest = Path(dest)
    assert isinstance(dest, Path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_archive = os.path.join(tmp_dir, archive)
        base_name = Path(archive).stem
        if ".tar" in base_name:
            base_name = Path(base_name).stem
        if url[-1] != "/":
            url += "/"
        download(url + archive, tmp_archive, progress=progress)
        extract(tmp_archive, tmp_dir, progress=progress)
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
            if dest.is_dir():
                assert force, f"Set force=True to replace destination {dest}"
                shutil.rmtree(dest)
            move(tmp_dir, dest)


def patch(path, cwd=None):
    raise NotImplementedError
