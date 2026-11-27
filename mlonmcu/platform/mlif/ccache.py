"""Filesystem scopes and child-process configuration for the MLIF ccache feature."""

import os
import shlex
import shutil
from pathlib import Path


def run_root(platform):
    """Normalize paths against the run, including both generated sources and MLIF."""
    if platform.run_dir is not None:
        return platform.run_dir
    if platform.build_dir is None:
        raise ValueError("Initialize the MLIF build directory before using ccache")
    return Path(platform.build_dir).resolve().parent


def get_ccache_definitions(platform, definitions):
    definitions = dict(definitions)
    for language in ("C", "CXX"):
        definitions[f"CMAKE_{language}_COMPILER_LAUNCHER"] = platform.ccache_exe
        key = f"EXTRA_CMAKE_{language}_FLAGS"
        flags = definitions.get(key, "")
        if isinstance(flags, (tuple, list)):
            flags = ";".join(flags)
        # Keep hash_dir enabled: normalize DWARF paths instead of returning
        # objects containing the first run's absolute debug directory.
        mapping = shlex.quote(f"-fdebug-prefix-map={run_root(platform)}=.")
        definitions[key] = f"{flags};{mapping}" if flags else mapping
    return definitions


def stage_model_sources(platform, src):
    """Place generated sources inside CMake's binary tree with stable object names.

    CMake embeds absolute paths into object/dependency target names for sources
    outside its source and binary trees. Those names contain run IDs and cannot
    be normalized by CCACHE_BASEDIR. A real copy also handles MLIF templates
    which resolve source symlinks with REALPATH.
    """
    src = Path(src).resolve()
    build_dir = Path(platform.build_dir).resolve()
    if src.is_relative_to(build_dir):
        return src
    dest = build_dir / "ccache-model"
    if dest.is_dir():
        shutil.rmtree(dest)

    # With stage_subdirs disabled the source directory is the run root, which
    # also contains this platform's build tree. Exclude it to avoid recursion.
    def ignore(directory, names):
        return [name for name in names if (Path(directory) / name).resolve() == build_dir]

    shutil.copytree(src, dest, ignore=ignore)
    return dest


def prepare_ccache_environment(platform, env):
    """Return an isolated child environment; never edit the user's configuration."""
    env = env.copy()
    if shutil.which(platform.ccache_exe, path=env.get("PATH")) is None:
        raise FileNotFoundError(
            f"ccache executable not found: {platform.ccache_exe}. Install ccache or set ccache.exe."
        )
    mode = platform.ccache_mode
    if mode != "global":
        # Managed scopes must not inherit a user's cache, config, remote storage,
        # or other CCACHE_* settings. All settings come from their own config.
        env = {key: value for key, value in env.items() if not key.startswith("CCACHE_")}
        if mode == "workspace":
            workspace = platform.workspace_dir or os.environ.get("MLONMCU_HOME")
            if workspace is None:
                raise ValueError("ccache workspace mode requires an MLonMCU context or MLONMCU_HOME")
            cache_dir = Path(workspace).resolve() / "cache" / "ccache"
        else:
            if platform.session_dir is None:
                raise ValueError(
                    "ccache session mode requires a session; use global/workspace mode for standalone MLIF"
                )
            cache_dir = platform.session_dir / "ccache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        config_path = cache_dir / "ccache.conf"
        # Exclusive creation also works when multiple runs start concurrently.
        try:
            with config_path.open("x"):
                pass
        except FileExistsError:
            pass
        env["CCACHE_DIR"] = str(cache_dir)
        env["CCACHE_CONFIGPATH"] = str(config_path)

    # A workspace/session base would leave run IDs in paths relative to CWD.
    # The run root rewrites /runs/0/build/foo.c and /runs/1/build/foo.c alike.
    env["CCACHE_BASEDIR"] = str(run_root(platform))
    for value, key in (
        (platform.ccache_max_size, "CCACHE_MAXSIZE"),
        (platform.ccache_max_files, "CCACHE_MAXFILES"),
        (platform.ccache_compiler_check, "CCACHE_COMPILERCHECK"),
    ):
        if value is not None:
            if key == "CCACHE_MAXFILES" and value < 0:
                raise ValueError("ccache.max_files must be nonnegative")
            env[key] = str(value)
    if platform.ccache_compression is not None:
        env.pop("CCACHE_COMPRESS", None)
        env.pop("CCACHE_NOCOMPRESS", None)
        env["CCACHE_COMPRESS" if platform.ccache_compression else "CCACHE_NOCOMPRESS"] = "1"
    return env
