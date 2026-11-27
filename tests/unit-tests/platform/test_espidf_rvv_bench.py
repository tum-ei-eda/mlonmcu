from pathlib import Path
import shutil
import subprocess

import pytest

from mlonmcu.models.frontend import RVVBenchFrontend
from mlonmcu.models.model import RVVBenchProgram
from mlonmcu.platform.espidf.espidf import EspIdfPlatform, get_project_template
from mlonmcu.platform.espidf.espidf_target import create_espidf_platform_target


@pytest.mark.parametrize("benchmark", ["instructions_scalar", "instructions_pie"])
def test_scalar_project_selection(tmp_path, monkeypatch, benchmark):
    frontend = RVVBenchFrontend(config={"mlif.src_dir": str(tmp_path / "mlif")})
    model = RVVBenchProgram(benchmark)
    platform = EspIdfPlatform(config={"espidf.src_dir": tmp_path, "espidf.install_dir": tmp_path})
    frontend.add_platform_defs("espidf", platform.definitions)
    model.add_platform_defs("espidf", platform.definitions)
    target = create_espidf_platform_target("esp32p4", platform)(config={"esp32p4.revision": "1.3"})
    target.add_platform_defs("espidf", platform.definitions)
    platform.project_dir = tmp_path / "project"
    monkeypatch.setattr(platform, "check", lambda: None)
    calls = []
    monkeypatch.setattr(platform, "invoke_idf_exe", lambda *args, **kwargs: calls.append(args) or "")
    platform.prepare(target, tmp_path / "codegen")
    defaults = (platform.project_dir / "sdkconfig.defaults").read_text()
    assert f'CONFIG_RVV_BENCH_DIR="{tmp_path / "mlif/lib/rvv_bench"}"' in defaults
    assert f'CONFIG_RVV_BENCH_BENCHMARK="{benchmark}"' in defaults
    assert "CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y" in defaults
    assert (platform.project_dir / "main/target.h").is_file()
    assert not (platform.project_dir / "components/tvm-lib").exists()
    assert calls[0][-2:] == ("set-target", "esp32p4")


def test_mlif_selection_unchanged():
    frontend = RVVBenchFrontend()
    assert frontend.get_platform_config("mlif") == {"template": "rvv_bench"}
    assert RVVBenchProgram("instructions_rvv").get_platform_defs("mlif") == {"RVV_BENCH_BENCHMARK": "instructions_rvv"}


def test_missing_sources():
    with pytest.raises(ValueError, match="mlif.src_dir"):
        RVVBenchFrontend().get_platform_defs("espidf")


def test_vector_rejected_on_espidf():
    with pytest.raises(ValueError, match="only instructions_scalar"):
        RVVBenchProgram("instructions_rvv").get_platform_defs("espidf")


def test_pie_rejected_on_mlif():
    with pytest.raises(ValueError, match="only the ESP-IDF"):
        RVVBenchProgram("instructions_pie").get_platform_defs("mlif")


def test_pie_frontend_lookup():
    models = RVVBenchFrontend().lookup_models(["instructions_pie"])
    assert len(models) == 1
    assert models[0].get_platform_defs("espidf") == {"RVV_BENCH_BENCHMARK": "instructions_pie"}


def test_pie_rejects_new_isa_configuration(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed")
    component = Path(get_project_template("rvv_bench")) / "main" / "CMakeLists.txt"
    script = tmp_path / "revision.cmake"
    script.write_text(
        "set(CONFIG_IDF_TARGET_ESP32P4 y)\n"
        "set(CONFIG_RVV_BENCH_BENCHMARK instructions_pie)\n"
        "set(CONFIG_ESP32P4_SELECTS_REV_LESS_V3 n)\n"
        f'include("{component}")\n'
    )
    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "instructions_pie targets PIE v1" in result.stderr
