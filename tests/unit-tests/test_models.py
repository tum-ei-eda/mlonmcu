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
import mock
import pytest
import yaml
import re
from mlonmcu.environment.environment import PathConfig
from mlonmcu.models.lookup import print_summary

# def test_models_get_model_directories():
#     pass
#
# def test_models_find_metadata():
#     pass
#
# def test_models_list_model_subdir():  # TODO
#     pass
#
# def test_models_list_models():
#     pass
#
# def test_models_list_modelgroups():
#     pass
#
# def lookup_models_and_groups():
#     pass

# def test_models_print_paths():
#     pass
#
# def test_models_print_models():
#     pass
#
# def test_models_print_groups():
#     pass


def _create_fake_metadata(path, base="metadata", ext="yaml"):
    metadata_file = path / f"{base}.{ext}"
    dummy_data = {}
    with open(metadata_file.resolve(), "w") as handle:
        yaml.dump(dummy_data, handle)


def _create_fake_models(path, names, ext="tflite", with_subdir=True, with_metadata=False):
    for name in names:
        if with_subdir:
            path.mkdir(exist_ok=True)
            model_dir = path / name
        else:
            model_dir = path
        model_dir.mkdir()
        model_file = model_dir / f"{name}.{ext}"
        model_file.touch()
        if with_metadata:
            if with_subdir:
                _create_fake_metadata(model_dir)
            else:
                _create_fake_metadata(model_dir, base=name)


def _create_fake_modelgroups(path, data={}, base="groups", ext="yaml"):
    path.mkdir(exist_ok=True)
    groups_file = path / f"{base}.{ext}"

    with open(groups_file.resolve(), "w") as handle:
        yaml.dump(data, handle)


def _check_summary_output(
    out,
    expected_paths=[],
    expected_models=[],
    expected_groups={},
    expected_skipped=0,
    expected_metadata_count=0,
    expected_duplicates=0,
    expected_group_duplicates=0,
    detailed=False,
):
    expected_paths_str = ["    " + str(path) for path in expected_paths]
    expected_models_str = ["    " + name for name in expected_models]
    expected_groups_str = [
        "    " + name + " [{} models]".format(len(models)) for name, models in expected_groups.items()
    ]
    skipped_count = 0
    duplicates_count = 0
    metadata_count = 0
    for line in out.split("\n"):
        if "(skipped)" in line:
            skipped_count = skipped_count + 1
            line = line.replace(" (skipped)", "")
        match = re.compile(r".*\((\d*) duplicate[s]*\).*").search(line)
        if match:
            count = int(match.group(1))
            duplicates_count = duplicates_count + count
            line = line[: line.find(" (")]

        if detailed:
            if line.startswith("        "):
                if "Metadata: available" in line:
                    metadata_count = metadata_count + 1
                continue
        assert (
            line
            in ["Models Summary", "Paths:", "Models:", "Groups:", ""]
            + expected_paths_str
            + expected_models_str
            + expected_groups_str
        )
    assert skipped_count == expected_skipped
    assert duplicates_count == expected_duplicates + expected_group_duplicates
    if detailed:
        assert metadata_count == expected_metadata_count


@pytest.mark.parametrize("detailed", [False, True])
def test_models_print_summary(detailed, capsys, fake_context, fake_environment_directory, fake_config_home):
    # list empty
    with mock.patch.dict(fake_context.environment.paths, {"models": []}):
        print_summary(fake_context, detailed=detailed)
        out, err = capsys.readouterr()
        for line in out.split("\n"):
            assert line in ["Models Summary", "Paths:", "Models:", "Groups:", ""]

    env_models_dir = fake_environment_directory / "models"
    env_models_dir.mkdir()

    # single path, no models
    with mock.patch.dict(fake_context.environment.paths, {"models": [PathConfig(env_models_dir)]}):
        print_summary(fake_context, detailed=detailed)
        out, err = capsys.readouterr()
        _check_summary_output(
            out,
            detailed=detailed,
            expected_paths=[env_models_dir],
        )

    _create_fake_models(env_models_dir, ["model0"], with_metadata=True)
    _create_fake_models(env_models_dir, ["model1", "model2"])

    # single path, 3 models
    with mock.patch.dict(fake_context.environment.paths, {"models": [PathConfig(env_models_dir)]}):
        print_summary(fake_context, detailed=detailed)
        out, err = capsys.readouterr()
        _check_summary_output(
            out,
            detailed=detailed,
            expected_paths=[env_models_dir],
            expected_models=["model0", "model1", "model2"],
            expected_metadata_count=1,
        )

    user_models_dir = fake_config_home / "models"

    # two paths (1 skipped), 3 models
    with mock.patch.dict(
        fake_context.environment.paths, {"models": [PathConfig(env_models_dir), PathConfig(user_models_dir)]}
    ):
        print_summary(fake_context, detailed=detailed)
        out, err = capsys.readouterr()
        _check_summary_output(
            out,
            detailed=detailed,
            expected_paths=[env_models_dir, user_models_dir],
            expected_models=["model0", "model1", "model2"],
            expected_skipped=1,
            expected_metadata_count=1,
        )

    user_models_dir.mkdir()
    _create_fake_models(fake_config_home / "models", ["model2", "model3"])

    # two paths, 5 models, 1 duplicate
    with mock.patch.dict(
        fake_context.environment.paths,
        {"models": [PathConfig(fake_environment_directory / "models"), PathConfig(fake_config_home / "models")]},
    ):
        print_summary(fake_context, detailed=detailed)
        out, err = capsys.readouterr()
        _check_summary_output(
            out,
            detailed=detailed,
            expected_paths=[env_models_dir, user_models_dir],
            expected_models=["model0", "model1", "model2", "model3"],
            expected_metadata_count=1,
            expected_duplicates=1,
        )

    _create_fake_modelgroups(env_models_dir, {"mygroup": ["model0", "model2"]})

    # two paths, 5 models, 1 duplicate, 1 group
    with mock.patch.dict(
        fake_context.environment.paths,
        {"models": [PathConfig(fake_environment_directory / "models"), PathConfig(fake_config_home / "models")]},
    ):
        print_summary(fake_context, detailed=detailed)
        out, err = capsys.readouterr()
        print("OUT", out)
        _check_summary_output(
            out,
            detailed=detailed,
            expected_paths=[env_models_dir, user_models_dir],
            expected_models=["model0", "model1", "model2", "model3"],
            expected_groups={"mygroup": ["model0", "model2"]},
            expected_metadata_count=1,
            expected_duplicates=1,
        )
    # TODO: group name conflicts with modelname
    # TODO: duplicate groups
