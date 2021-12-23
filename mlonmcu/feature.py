
class Feature():
    def __init__(self, name, cfg={}):
        self.name = name
        self.cfg = cfg

    def __repr__(self):
        return type(self).__name__ + f"({self.name})"

class FrontendFeature(Feature):
    pass

class FrameworkFeature(Feature):
    pass

class BackendFeature(Feature):
    pass

class TargetFeature(Feature):
    pass

def lookup_features(name):
    print("ALL_FEATURES", ALL_FEATURES)
    assert name in ALL_FEATURES, f"Unknown feature: {name}"
    ret = [] # For a single feature-name, there can be multiple types of features (e.g. backend vs target) and we want to match all of them
    if name in FRONTEND_FEATURES:
        ret.append(FrontendFeature(name))
    if name in FRAMEWORK_FEATURES:
        ret.append(FrameworkFeature(name))
    if name in BACKEND_FEATURES:
        ret.append(BackendFeature(name))
    if name in TARGET_FEATURES:
        ret.append(TargetFeature(name))
    return ret

# Frontend features
TFLITE_FRONTEND_FEATURES = ["packing"]
FRONTEND_FEATURES = TFLITE_FRONTEND_FEATURES

# Framework features
TFLITE_FRAMEWORK_FEATURES = ["packing", "muriscvnn"]
TVM_FRAMEWORK_FEATURES = ["autotuning"]
FRAMEWORK_FEATURES = TFLITE_FRAMEWORK_FEATURES + TVM_FRAMEWORK_FEATURES

# Backend features
TFLITE_BACKEND_FEATURES = []
TVMAOT_BACKEND_FEATURES = ["unpacked_api"]
TVM_BACKEND_FEATURES = TVMAOT_BACKEND_FEATURES
BACKEND_FEATURES = TFLITE_BACKEND_FEATURES + TVM_BACKEND_FEATURES

# Traget features
TARGET_FEATURES = ["trace"]

ALL_FEATURES = FRONTEND_FEATURES + FRAMEWORK_FEATURES + BACKEND_FEATURES + TARGET_FEATURES
