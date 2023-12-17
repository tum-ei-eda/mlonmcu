import pytest

# from mlonmcu.session.run import RunStage

# from mlonmcu.artifact import ArtifactFormat
from mlonmcu.artifact import lookup_artifacts

from mlonmcu.testing.helpers import _test_frontend

# from mlonmcu.testing.helpers import _check_features, _init_run


# TODO: make sure that we use quant/float models and several different operators
DEFAULT_TFLITE_MODELS = ["sine_model"]
DEFAULT_ONNX_MODELS = ["onnx_mnist"]
DEFAULT_RELAY_MODELS = ["test_cnn"]
DEFAULT_PB_MODELS = ["mobilenet_v1_1_0_224"]
ALL_PB_MODELS = ["mobilenet_v1_1_0_224", "mobilenet_v1_1_0_224_quant", "mobilenet_v1_0_25_128"]
DEFAULT_PADDLE_MODELS = ["paddle_resnet50"]


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("feature_names", [[], ["tflite_visualize"]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_tflite(user_context, model_name, models_dir, feature_names, config):
    _, artifacts = _test_frontend("tflite", user_context, model_name, models_dir, feature_names, config)


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
# @pytest.mark.parametrize("feature_names", [[], ["tflite_visualize"]])
@pytest.mark.parametrize("feature_names", [["visualize"]])  # TODO: rename
@pytest.mark.parametrize("config", [{}])
def test_feature_tflite_visualize(user_context, model_name, models_dir, feature_names, config):
    _, artifacts = _test_frontend("tflite", user_context, model_name, models_dir, feature_names, config)
    assert len(lookup_artifacts(artifacts, name="tflite_visualize.html")) == 1


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_ONNX_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_onnx(user_context, model_name, models_dir, feature_names, config):
    _test_frontend("onnx", user_context, model_name, models_dir, feature_names, config)


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_RELAY_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_relay(user_context, model_name, models_dir, feature_names, config):
    _test_frontend("relay", user_context, model_name, models_dir, feature_names, config)


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_RELAY_MODELS)
@pytest.mark.parametrize("feature_names", [["relayviz"]])
@pytest.mark.parametrize("config", [{"relayviz.plotter": "term"}, {"relayviz.plotter": "dot"}])
def test_feature_relayviz(user_context, model_name, models_dir, feature_names, config):
    _, artifacts = _test_frontend("relay", user_context, model_name, models_dir, feature_names, config)
    ext = "txt" if config["relayviz.plotter"] == "term" else "pdf"
    assert len(lookup_artifacts(artifacts, name=f"relayviz.{ext}")) == 1


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_PB_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_pb(user_context, model_name, models_dir, feature_names, config):
    _test_frontend("pb", user_context, model_name, models_dir, feature_names, config)


@pytest.mark.skip(reason="unimplemented")
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_PADDLE_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_paddle(user_context, model_name, models_dir, feature_names, config):
    _test_frontend("paddle", user_context, model_name, models_dir, feature_names, config)
