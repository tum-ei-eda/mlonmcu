"""Optional model downloader behavior, using local scripts without network access."""

import io

import numpy as np
import pytest

from mlonmcu.feature.features import get_available_features
from mlonmcu.models.data_utils import ensure_model_data
from mlonmcu.models.frontend import TfLiteFrontend
from mlonmcu.models.model import Model


@pytest.fixture
def downloadable_model(tmp_path):
    (tmp_path / "model.tflite").write_bytes(b"model")
    np.array([[1]], dtype="float32").tofile(tmp_path / "source.bin")
    (tmp_path / "download_data.sh").write_text(
        "#!/bin/bash\n"
        "test -f model.tflite\n"
        "test -d input\n"
        "test -d output\n"
        "printf 'run\\n' >> calls\n"
        "cp source.bin input/0.bin\n"
        "cp source.bin output/0.bin\n"
        "printf 'i,label_idx\\n0,0\\n' > labels.csv\n"
    )
    model = Model("model", tmp_path / "model.tflite")
    model.metadata = {
        "network_parameters": {
            "input_nodes": [{"name": "x", "shape": [1, 1], "dtype": "float32", "example_input": {"path": "input"}}],
            "output_nodes": [
                {
                    "name": "y",
                    "shape": [1, 1],
                    "dtype": "float32",
                    "test_output_path": "output",
                    "test_labels_file": "labels.csv",
                }
            ],
        }
    }
    return model


def frontend(download=False, mode="file"):
    config = {"gen_data.number": 1, "gen_ref_data.mode": mode, "gen_ref_labels.mode": mode}
    registry = get_available_features()
    features = [registry[name](config=config) for name in ["gen_data", "gen_ref_data", "gen_ref_labels"]]
    return TfLiteFrontend(features=features, config={"tflite.download_data": download})


def test_download_is_opt_in(downloadable_model, tmp_path):
    with pytest.raises(AssertionError, match="Input data path does not exist"):
        frontend().process_metadata(downloadable_model, cfg={})
    assert not (tmp_path / "calls").exists()
    assert not (tmp_path / "input").exists()


def test_downloader_populates_artifacts_and_is_not_repeated(downloadable_model, tmp_path):
    loader = frontend(download="1")
    for _ in range(2):
        artifacts = {artifact.name: artifact for artifact in loader.process_metadata(downloadable_model, cfg={})}
        for name in ["inputs.npy", "outputs_ref.npy", "labels_ref.npy"]:
            assert len(np.load(io.BytesIO(artifacts[name].raw), allow_pickle=True)) == 1
    assert (tmp_path / "calls").read_text() == "run\n"


@pytest.mark.parametrize("missing", ["input", "output", "labels"])
def test_empty_or_partial_data_triggers_downloader(downloadable_model, tmp_path, missing):
    loader = frontend(download=True)
    loader.process_metadata(downloadable_model, cfg={})
    target = {"input": "input/0.bin", "output": "output/0.bin", "labels": "labels.csv"}[missing]
    (tmp_path / target).unlink()
    loader.process_metadata(downloadable_model, cfg={})
    assert (tmp_path / "calls").read_text() == "run\nrun\n"


def test_model_references_do_not_download_unused_files(downloadable_model, tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input/0.bin").write_bytes((tmp_path / "source.bin").read_bytes())
    loader = frontend(download=True, mode="model")
    loader.inference = lambda model, data, **kwargs: {"y": data["x"]}
    loader.process_metadata(downloadable_model, cfg={})
    assert not (tmp_path / "calls").exists()
    assert not (tmp_path / "output").exists()


def test_missing_script_keeps_existing_error(downloadable_model, tmp_path):
    (tmp_path / "download_data.sh").unlink()
    with pytest.raises(AssertionError, match="Input data path does not exist"):
        frontend(download=True).process_metadata(downloadable_model, cfg={})


def test_downloader_failure_is_propagated(tmp_path):
    (tmp_path / "download_data.sh").write_text("exit 7\n")
    with pytest.raises(AssertionError, match="non-zero exit code 7"):
        ensure_model_data(tmp_path, input_paths=[tmp_path / "input"], number=1)


def test_successful_script_must_provide_data(tmp_path):
    (tmp_path / "download_data.sh").write_text("true\n")
    with pytest.raises(RuntimeError, match="did not provide required validation data"):
        ensure_model_data(tmp_path, input_paths=[tmp_path / "input"], number=1)


def test_requested_samples_and_tensor_indices_are_checked(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "0_0.bin").write_bytes(b"first")
    (tmp_path / "download_data.sh").write_text(
        "printf run > calls\n"
        "printf second > input/0_1.bin\n"
        "printf third > input/1_0.bin\n"
        "printf fourth > input/1_1.bin\n"
    )
    ensure_model_data(tmp_path, input_paths=[inputs, inputs], number=2)
    assert (tmp_path / "calls").read_text() == "run"


def test_manifest_limits_download_to_requested_samples(tmp_path):
    from mlonmcu.models.data_utils import write_model_data_manifest

    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "0.bin").write_bytes(b"zero")
    (inputs / "1.bin").write_bytes(b"one")
    manifest = write_model_data_manifest(tmp_path, ["input"])
    assert manifest["sample_count"] == 2
    (inputs / "1.bin").unlink()
    (tmp_path / "download_data.sh").write_text("printf run > calls\nprintf one > input/1.bin\n")
    assert ensure_model_data(tmp_path, input_paths=[inputs], number=1) == 1
    assert not (tmp_path / "calls").exists()
    # Requesting more than the full dataset is capped by the manifest.
    assert ensure_model_data(tmp_path, input_paths=[inputs], number=100) == 2
    assert (tmp_path / "calls").read_text() == "run"


def test_manifest_hashes_detect_and_restore_corruption(tmp_path):
    from mlonmcu.models.data_utils import write_model_data_manifest

    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "0.bin").write_bytes(b"original")
    write_model_data_manifest(tmp_path, ["input"])
    (inputs / "0.bin").write_bytes(b"modified")
    (tmp_path / "download_data.sh").write_text("printf original > input/0.bin\nprintf run > calls\n")
    ensure_model_data(tmp_path, input_paths=[inputs], number=1)
    assert (inputs / "0.bin").read_bytes() == b"original"
    assert (tmp_path / "calls").read_text() == "run"


def test_manifest_cap_is_respected_by_frontend(downloadable_model, tmp_path):
    from mlonmcu.models.data_utils import write_model_data_manifest

    loader = frontend(download=True)
    loader.process_metadata(downloadable_model, cfg={})
    write_model_data_manifest(tmp_path, ["input", "output", "labels.csv"])
    # Extra local files must not be loaded without being checked against the manifest.
    for directory in ["input", "output"]:
        (tmp_path / directory / "1.bin").write_bytes((tmp_path / "source.bin").read_bytes())
    loader.config["gen_data_number"] = 10
    artifacts = {artifact.name: artifact for artifact in loader.process_metadata(downloadable_model, cfg={})}
    assert len(np.load(io.BytesIO(artifacts["inputs.npy"].raw), allow_pickle=True)) == 1
    assert (tmp_path / "calls").read_text() == "run\n"


def test_manifest_generator_refuses_to_replace_existing_manifest(tmp_path):
    from mlonmcu.models.data_utils import write_model_data_manifest

    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "0.bin").write_bytes(b"zero")
    write_model_data_manifest(tmp_path, ["input"])
    with pytest.raises(FileExistsError):
        write_model_data_manifest(tmp_path, ["input"])


def test_parallel_download_requests_run_script_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    inputs = tmp_path / "input"
    (tmp_path / "download_data.sh").write_text("printf run >> calls\nprintf zero > input/0.bin\n")
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(ensure_model_data, tmp_path, input_paths=[inputs], number=1) for _ in range(2)]
        assert [job.result() for job in jobs] == [1, 1]
    assert (tmp_path / "calls").read_text() == "run"


def test_custom_paths_do_not_download_default_dataset(downloadable_model, tmp_path):
    inputs = tmp_path / "custom_input"
    outputs = tmp_path / "custom_output"
    for directory in [inputs, outputs]:
        directory.mkdir()
        (directory / "0.bin").write_bytes((tmp_path / "source.bin").read_bytes())
    labels = tmp_path / "custom_labels.csv"
    labels.write_text("i,label_idx\n0,0\n")
    downloadable_model.config.update(inputs_path=inputs, outputs_path=outputs, output_labels_path=labels)
    frontend(download=True).process_metadata(downloadable_model, cfg={})
    assert not (tmp_path / "calls").exists()


def test_manifest_requires_hashes_for_selected_files(tmp_path):
    import json

    (tmp_path / "data_manifest.json").write_text(json.dumps({"version": 1, "sample_count": 1, "files": {}}))
    with pytest.raises(ValueError, match="does not describe all requested"):
        ensure_model_data(tmp_path, input_paths=[tmp_path / "input"], number=1)


def test_corruption_still_present_after_download_is_an_error(tmp_path):
    from mlonmcu.models.data_utils import write_model_data_manifest

    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "0.bin").write_bytes(b"original")
    write_model_data_manifest(tmp_path, ["input"])
    (inputs / "0.bin").write_bytes(b"corrupt")
    (tmp_path / "download_data.sh").write_text("true\n")
    with pytest.raises(RuntimeError, match="did not provide required validation data"):
        ensure_model_data(tmp_path, input_paths=[inputs], number=1)
