"""ESP-IDF hardware revision selection and sdkconfig generation."""

import pytest

from mlonmcu.platform.espidf.espidf import EspIdfPlatform
from mlonmcu.platform.espidf.espidf_target import create_espidf_platform_target


@pytest.mark.parametrize(
    "revision,minimum,legacy",
    [("0.0", 0, True), ("0.1", 1, True), ("1.3", 100, True), ("3.0", 300, False), ("3.1", 301, False)],
)
def test_revision_sdkconfig(tmp_path, monkeypatch, revision, minimum, legacy):
    platform = EspIdfPlatform(config={"espidf.src_dir": tmp_path, "espidf.install_dir": tmp_path})
    target = create_espidf_platform_target("esp32p4", platform)(config={"esp32p4.revision": revision})
    target.add_platform_defs("espidf", platform.definitions)
    platform.project_dir = tmp_path / "project"
    monkeypatch.setattr(platform, "check", lambda: None)
    monkeypatch.setattr(platform, "invoke_idf_exe", lambda *args, **kwargs: "")
    platform.prepare(target, tmp_path / "codegen")
    defaults = (platform.project_dir / "sdkconfig.defaults").read_text()
    assert f"CONFIG_ESP32P4_SELECTS_REV_LESS_V3={'y' if legacy else 'n'}\n" in defaults
    assert f"CONFIG_ESP32P4_REV_MIN_{minimum}=y\n" in defaults


def test_revision_optional():
    target = create_espidf_platform_target("esp32p4", None)()
    assert target.get_platform_defs("espidf") == {}


@pytest.mark.parametrize("revision", ["1", "1.100", "2.0", "4.0", "-1.0", "invalid"])
def test_revision_invalid(revision):
    target = create_espidf_platform_target("esp32p4", None)(config={"esp32p4.revision": revision})
    with pytest.raises(ValueError, match="esp32p4.revision"):
        target.get_platform_defs("espidf")


def test_revision_unsupported_target():
    target = create_espidf_platform_target("esp32c3", None)(config={"esp32c3.revision": "1.0"})
    with pytest.raises(ValueError, match="only for esp32p4"):
        target.get_platform_defs("espidf")
