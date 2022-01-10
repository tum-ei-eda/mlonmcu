import mock
import pytest
from mlonmcu.environment.environment import PathConfig
from mlonmcu.models.lookup import print_summary

def test_models_get_model_directories():
    pass

def test_models_find_metadata():
    pass

def test_models_list_model_subdir():  # TODO
    pass

def test_models_list_models():
    pass

def test_models_list_modelgroups():
    pass

def test_models_print_paths():
    pass

def test_models_print_models():
    pass

def test_models_print_groups():
    pass

@pytest.mark.parametrize("detailed", [False, True])
def test_models_print_summary(detailed, fake_context, fake_environment_directory, fake_config_home):
    # list empty
    with mock.patch.dict(fake_context.environment.paths, {"models": []}) as mocked:
        pass
    # single
    with mock.patch.dict(fake_context.environment.paths, {"models": [PathConfig(fake_environment_directory / "models")]}) as mocked:
        # create dir
        pass
    # multiple
    with mock.patch.dict(fake_context.environment.paths, {"models": [PathConfig(fake_environment_directory / "models"), PathConfig(fake_config_home / "models")]}) as mocked:
        # create dirs
        print_summary(fake_context, detailed=detailed)
    # skipped
    with mock.patch.dict(fake_context.environment.paths, {"models": [PathConfig(fake_environment_directory / "models"), PathConfig(fake_config_home / "models")]}) as mocked:
        # create 1 dir
        pass
