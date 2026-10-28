"""Helpers for managed workspace Python virtual environments."""

import json
import os
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


def get_venv_bin_dir(venv_dir):
    return Path(venv_dir) / ("Scripts" if os.name == "nt" else "bin")


def get_venv_python(venv_dir):
    return get_venv_bin_dir(venv_dir) / ("python.exe" if os.name == "nt" else "python")


def get_install_requirement(extras):
    """Describe the currently running MLonMCU installation for pip."""
    suffix = f"[{','.join(extras)}]" if extras else ""
    dist = metadata.distribution("mlonmcu")
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        data = json.loads(direct_url)
        parsed = urlparse(data["url"])
        if parsed.scheme == "file":
            source = unquote(parsed.path)
            requirement = f"{source}{suffix}"
            if data.get("dir_info", {}).get("editable", False):
                return ["-e", requirement]
            return [requirement]
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return ["-e", f"{source_root}{suffix}"]
    return [f"mlonmcu{suffix}=={dist.version}"]


def get_workspace_process_env(venv_dir, base_env=None, paths_file=None):
    """Build child-process environment equivalent to activating a workspace."""
    env = dict(os.environ if base_env is None else base_env)
    path = env.get("PATH", "")
    if paths_file and Path(paths_file).is_file():
        for line in Path(paths_file).read_text(encoding="utf-8").splitlines():
            if line.startswith("export PATH="):
                path = line.removeprefix("export PATH=").replace("$PATH", path)
                break
    env["PATH"] = os.pathsep.join(entry for entry in (str(get_venv_bin_dir(venv_dir)), path) if entry)
    env["VIRTUAL_ENV"] = str(venv_dir)
    return env
