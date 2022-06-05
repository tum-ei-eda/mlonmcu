import itertools
import logging

# import mlonmcu
import mlonmcu.context
from mlonmcu.session.run import RunStage
from mlonmcu.feature.features import (
    get_available_features,
)
from mlonmcu.config import resolve_required_config
from mlonmcu.logging import get_logger, set_log_level

logger = get_logger()
set_log_level(logging.DEBUG)

TRIALS = 100
PARALLEL = 1
PROGRESS = True

FRONTEND = "tflite"
TARGET = "etiss_pulpino"
PLATFORM = "microtvm"
# BACKEND = "tvmaot"
# BACKEND = "tvmllvm"
BACKEND = "tvmrt"

STAGE = RunStage.TUNE

MODELS = [
    "sine_model",
    # "magic_wand",
    # "micro_speech",
    # "cifar10",
    # "simple_mnist",
    # "aww",
    # "vww",
    # "resnet",
    # "toycar",
]

# DEFAULT_FEATURES = ["usmp", "unpacked_api", "autotune"]
DEFAULT_FEATURES = ["autotune"]

FEATURES = [
    (),
    ("disable_legalize",),
]

DEFAULT_CONFIG = {
    # "usmp.algorighm": "hill_climb",
    "tvmaot.print_outputs": True,
}

TARGET_CONFIGS = [
    {f"{BACKEND}.target_device": "riscv_cpu", "tvmaot.target_model": "etissvp"},  # default
    {f"{BACKEND}.target_device": "arm_cpu", "tvmaot.target_modeo": "etissvp"},  # arm_schedules
]

LAYOUT_CONFIGS = [
    # {"tvmaot.desired_layout": None},
    {f"{BACKEND}.desired_layout": "NHWC"},
    {f"{BACKEND}.desired_layout": "NCHW"},
]

def merge_dicts(d):
    return [{k: v for y in x for k, v in y.items()} for x in d]

CONFIGS = merge_dicts(list(itertools.product(TARGET_CONFIGS, LAYOUT_CONFIGS)))

FEATURES = [ set(DEFAULT_FEATURES + list(x)) for x in FEATURES]
CONFIGS = [ {**DEFAULT_CONFIG, **x} for x in CONFIGS]

print("FEATURES", FEATURES, len(FEATURES))
print("CONFIGS", CONFIGS, len(CONFIGS))

FEATURES_CONFIG = list(itertools.product(FEATURES, CONFIGS))

print("FEATURES_CONFIG", FEATURES_CONFIG, len(FEATURES_CONFIG))

def init_features_by_name(names, config, context=None):
    features = []
    for name in names:
        available_features = get_available_features(feature_name=name)
        for feature_cls in available_features:
            required_keys = feature_cls.REQUIRED
            if len(required_keys) > 0:
                assert context is not None
                config.update(
                    resolve_required_config(
                        required_keys,
                        features=features,  # The order the features are provided is important here!
                        config=config,
                        cache=context.cache,
                    )
                )
            feature_inst = feature_cls(config=config)
            features.append(feature_inst)
        return features

with mlonmcu.context.MlonMcuContext() as context:
    session = context.create_session()
    for model_name in MODELS:
        for features_names, config in FEATURES_CONFIG:
            features = init_features_by_name(features_names, config, context=context)
            run = session.create_run(features=features, config=config)
            run.add_frontend_by_name(FRONTEND, context=context)
            run.add_model_by_name(model_name, context=context)
            run.add_backend_by_name(BACKEND, context=context)
            # run.add_target_by_name(TARGET, context=context)
            run.add_platform_by_name(PLATFORM, context=context)
    print("sesion.runs", session.runs, len(session.runs))
    session.process_runs(until=STAGE, num_workers=PARALLEL, progress=PROGRESS, context=context)
    report = session.get_reports()
