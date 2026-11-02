"""Cache scope, lifecycle, and real compiler reuse across MLIF run directories."""

import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from mlonmcu.feature.features import get_available_features
from mlonmcu.platform.mlif.mlif import MlifPlatform
from mlonmcu.session.run import Run, RunInitializer
from mlonmcu.session.session import Session

Ccache = get_available_features(feature_name="ccache")["ccache"]


def platform_for(tmp_path, mode="session", **options):
    feature = Ccache(config={"ccache.mode": mode, **{f"ccache.{k}": v for k, v in options.items()}})
    platform = MlifPlatform(features=[feature], config={"mlif.src_dir": tmp_path, "mlif.num_threads": 2})
    platform.set_context(workspace=tmp_path / "workspace", run_dir=tmp_path / "runs/0", session_dir=tmp_path)
    platform.build_dir = platform.run_dir / "mlif"
    platform.build_dir.mkdir(parents=True)
    return platform


@pytest.mark.parametrize("mode", ["global", "workspace", "session"])
def test_scopes_and_settings(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: "/usr/bin/ccache")
    inherited = {
        "CCACHE_DIR": str(tmp_path / "user-cache"),
        "CCACHE_CONFIGPATH": str(tmp_path / "user.conf"),
        "CCACHE_SLOPPINESS": "time_macros",
        "CCACHE_NOCOMPRESS": "1",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    before = os.environ.copy()
    platform = platform_for(tmp_path, mode, max_size="2G", max_files="1000", compression=True, compiler_check="content")
    env = platform.prepare_environment()
    assert os.environ == before
    assert env["CCACHE_BASEDIR"] == str(platform.run_dir)
    assert env["CCACHE_MAXSIZE"] == "2G"
    assert env["CCACHE_MAXFILES"] == "1000"
    assert env["CCACHE_COMPILERCHECK"] == "content"
    assert env["CCACHE_COMPRESS"] == "1"
    assert "CCACHE_NOCOMPRESS" not in env
    if mode == "global":
        assert all(env[key] == value for key, value in inherited.items() if key != "CCACHE_NOCOMPRESS")
        assert not (tmp_path / "user-cache").exists()
    else:
        expected = tmp_path / ("workspace/cache/ccache" if mode == "workspace" else "ccache")
        assert env["CCACHE_DIR"] == str(expected)
        assert env["CCACHE_CONFIGPATH"] == str(expected / "ccache.conf")
        assert "CCACHE_SLOPPINESS" not in env
        # Existing scope configuration survives later runs/configure calls.
        (expected / "ccache.conf").write_text("max_size = 3G\n")
        platform.prepare_environment()
        assert (expected / "ccache.conf").read_text() == "max_size = 3G\n"


def test_registration_disabled_and_invalid_config(tmp_path, monkeypatch):
    assert get_available_features(feature_name="ccache")["ccache"] is Ccache
    assert Ccache().mode == "session"
    with pytest.raises(AssertionError):
        Ccache(config={"ccache.mode": "invalid"}).get_platform_config("mlif")
    monkeypatch.setenv("CCACHE_DIR", "/unchanged")
    platform = MlifPlatform(features=[Ccache(config={"ccache.enabled": False})], config={"mlif.src_dir": tmp_path})
    assert not platform.ccache
    assert "CMAKE_C_COMPILER_LAUNCHER" not in platform.get_definitions()
    assert platform.prepare_environment()["CCACHE_DIR"] == "/unchanged"
    platform = platform_for(tmp_path, exe="missing-ccache-executable")
    with pytest.raises(FileNotFoundError, match="Install ccache"):
        platform.prepare_environment()
    platform.config["ccache_exe"] = "ccache"
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: "/usr/bin/ccache")
    platform.session_dir = None
    with pytest.raises(ValueError, match="requires a session"):
        platform.prepare_environment()


def test_initializer_binds_real_context_paths(tmp_path):
    # Workspace/session paths need not be related, or use the default names.
    workspace = tmp_path / "home"
    context = SimpleNamespace(
        cache={},
        environment=SimpleNamespace(home=workspace, flags=[], has_platform=lambda name: name == "mlif"),
    )
    init = RunInitializer(
        idx=0, platform_names=["mlif"], feature_names=["ccache"], config={"mlif.src_dir": str(tmp_path)}
    )
    with Session(dest=tmp_path / "custom-session") as session:
        run = init.realize(context=context)
        run.init_directory(parent=session.runs_dir)
        platform = run.platforms[0]
        assert platform.workspace_dir == workspace
        assert platform.run_dir == session.runs_dir / "0"
        assert platform.session_dir == session.dir
        # Only paths are retained, so process workers can pickle the platform.
        import pickle

        roots = (platform.workspace_dir, platform.run_dir, platform.session_dir)
        assert pickle.loads(pickle.dumps(roots)) == roots


@pytest.mark.parametrize("cleanup", [False, True])
def test_session_cleanup_retains_other_scopes(tmp_path, cleanup):
    workspace_cache = tmp_path / "cache/ccache"
    workspace_cache.mkdir(parents=True)
    (workspace_cache / "keep").touch()
    with pytest.raises(RuntimeError, match="test failure"):
        with Session(dest=tmp_path / "session", config={"session.ccache_cleanup": cleanup}) as session:
            session_cache = session.dir / "ccache"
            session_cache.mkdir()
            (session_cache / "entry").touch()
            raise RuntimeError("test failure")
    assert session_cache.exists() is not cleanup
    assert (workspace_cache / "keep").exists()


def stats(env):
    output = subprocess.check_output(["ccache", "--print-stats"], env=env, text=True)
    return {key: int(value) for key, value in (line.split() for line in output.splitlines())}


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("mode", ["global", "workspace", "session"])
def test_real_mlif_compile_reuses_cache_and_invalidates_changed_header(tmp_path, monkeypatch, debug, flat, mode):
    for exe in ("ccache", "cmake", "make", "gcc", "g++"):
        if shutil.which(exe) is None:
            pytest.skip(f"{exe} unavailable")
    # Exercise the actual MLIF configure/compile API with a tiny C/C++ project.
    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(ccache_test C CXX)\n"
        "foreach(lang C CXX)\n"
        "  foreach(flag IN LISTS EXTRA_CMAKE_${lang}_FLAGS)\n"
        '    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:${lang}>:${flag}>")\n'
        "  endforeach()\n"
        "endforeach()\n"
        'add_executable(generic_mlonmcu "${SRC_DIR}/main.c" "${SRC_DIR}/helper.cc")\n'
        'target_include_directories(generic_mlonmcu PRIVATE "${SRC_DIR}")\n'
    )
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "wrong-user-cache"))
    user_config = tmp_path / "user-ccache.conf"
    user_config.write_text("max_size = 10M\n")
    monkeypatch.setenv("CCACHE_CONFIGPATH", str(user_config))
    session = Session(dest=tmp_path / "session")
    with session:
        previous = None
        for idx in range(3):
            platform = MlifPlatform(
                features=[Ccache(config={"ccache.mode": mode})],
                config={"mlif.src_dir": str(project), "mlif.num_threads": 2, "mlif.debug": debug},
            )
            platform.set_context(workspace=tmp_path / "workspace")
            run = Run(idx=idx)
            run.add_platform(platform)
            run.init_directory(parent=session.runs_dir)
            src = run.dir if flat else run.dir / "build"
            src.mkdir(exist_ok=True)
            (src / "main.c").write_text('#include <stdio.h>\nint helper(void);\nint main(){printf("%d",helper());}\n')
            (src / "helper.cc").write_text('#include "value.h"\nextern "C" int helper(){return VALUE;}\n')
            (src / "value.h").write_text(f"#define VALUE {43 if idx == 2 else 42}\n")
            first_defs = platform.get_definitions()
            assert platform.get_definitions() == first_defs
            platform.compile(None, src=src)
            result = subprocess.check_output([str(platform.build_dir / "generic_mlonmcu")], text=True)
            assert result == ("43" if idx == 2 else "42")
            env = platform.prepare_environment()
            current = stats(env)
            if idx == 1:
                assert current["cache_miss"] == previous["cache_miss"]
                hits = sum(current[key] - previous[key] for key in ("direct_cache_hit", "preprocessed_cache_hit"))
                assert hits >= 2  # Both C and C++ compilation reused across /runs/{0,1}.
            elif idx == 2:
                assert current["cache_miss"] > previous["cache_miss"]
            previous = current
    assert (tmp_path / "wrong-user-cache").exists() is (mode == "global")
    assert user_config.read_text() == "max_size = 10M\n"


@pytest.mark.parametrize("mode", ["workspace", "session"])
def test_managed_cache_sharing_between_sessions(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: "/usr/bin/ccache")
    platform = platform_for(tmp_path, mode)
    first = platform.prepare_environment()
    platform.set_context(run_dir=tmp_path / "other-session/runs/0", session_dir=tmp_path / "other-session")
    second = platform.prepare_environment()
    assert (first["CCACHE_DIR"] == second["CCACHE_DIR"]) is (mode == "workspace")
    assert (first["CCACHE_CONFIGPATH"] == second["CCACHE_CONFIGPATH"]) is (mode == "workspace")
