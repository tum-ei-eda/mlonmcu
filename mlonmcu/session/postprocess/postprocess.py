from mlonmcu.config import filter_config


class Postprocess:

    FEATURES = []

    DEFAULTS = {}

    REQUIRED = []

    def __init__(self, name, config=None, features=None):
        self.name = name
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.REQUIRED)

    def process_features(self, features):
        # Currently there is no support for postprocess features
        return []
        # if features is None:
        #     return []
        # features = get_matching_features(features, FeatureType.POSTPROCESS)
        # for feature in features:
        #     assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
        #     feature.add_target_config(self.name, self.config)
        # return features


class SessionPostprocess(Postprocess):
    def post_session(self, report):
        pass


class RunPostprocess(Postprocess):
    def post_run(self, report):
        pass
