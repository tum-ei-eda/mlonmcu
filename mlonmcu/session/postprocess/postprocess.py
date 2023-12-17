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
"""Definitions of base classes for MLonMCU postprocesses."""
from mlonmcu.config import filter_config


class Postprocess:
    """Abstract postprocess."""

    FEATURES = set()

    DEFAULTS = {}

    REQUIRED = set()
    OPTIONAL = set()

    def __init__(self, name, config=None, features=None):
        self.name = name
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.OPTIONAL, self.REQUIRED)

    def process_features(self, features):
        """Utility which handles postprocess_features."""
        # Currently there is no support for postprocess features (FIXME)
        # if features is None:
        return []
        # features = get_matching_features(features, FeatureType.POSTPROCESS)
        # for feature in features:
        #     assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
        #     feature.add_target_config(self.name, self.config)
        # return features


class SessionPostprocess(Postprocess):
    """Session postprocess which is applied to multiple runs at the end of a session. (multi-row)"""

    def post_session(self, report):
        """Called at the end of a session."""


class RunPostprocess(Postprocess):
    """Run postprocess which is applied to a single run."""

    def post_run(self, report, artifacts):
        """Called at the end of a run."""
