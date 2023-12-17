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
import os
import time
import argparse
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, List

from mlonmcu.cli.helper.parse import extract_feature_names, extract_config
from mlonmcu.feature.type import FeatureType
from mlonmcu.config import filter_config
from mlonmcu.feature.features import get_matching_features
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger
from mlonmcu.target.metrics import Metrics

logger = get_logger()


class Backend(ABC):
    name = None

    FEATURES = set()
    DEFAULTS = {}
    REQUIRED = set()
    OPTIONAL = set()

    def __init__(
        self,
        framework="",
        features=None,
        config=None,
    ):
        self.framework = framework
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.OPTIONAL, self.REQUIRED)
        self.artifacts = []
        self.supported_fmts = []
        self.tuner = None

    def __repr__(self):
        name = type(self).name
        return f"Backend({name})"

    def process_features(self, features):
        # Filter out non-backend features
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.BACKEND)
        for feature in features:
            assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
            feature.used = True
            feature.add_backend_config(self.name, self.config)
        return features

    @abstractmethod
    def load_model(self, model, input_shapes=None, output_shapes=None, input_types=None, output_types=None):
        pass

    @abstractmethod
    def generate(self) -> Tuple[dict, dict]:
        return {}, {}

    def generate_artifacts(self) -> List[Artifact]:
        start_time = time.time()
        artifacts, metrics = self.generate()
        # TODO: do something with out?
        end_time = time.time()
        diff = end_time - start_time
        if len(metrics) == 0:
            metrics = {"default": Metrics()}
        for name, metrics_ in metrics.items():
            if name == "default":
                metrics_.add("Build Stage Time [s]", diff, True)
            content = metrics_.to_csv(include_optional=True)  # TODO: store df instead?
            artifact = Artifact("build_metrics.csv", content=content, fmt=ArtifactFormat.TEXT, flags=["metrics"])
            if name not in artifacts:
                artifacts[name] = []
            artifacts[name].append(artifact)
        self.artifacts = artifacts
        return artifacts

    @property
    def has_tuner(self):
        return self.tuner is not None

    @property
    def needs_target(self):
        return False

    def set_tuning_records(self, filepath):
        if not self.has_tuner:
            raise NotImplementedError("Backend does not support autotuning")

    def export_artifacts(self, path):
        assert len(self.artifacts) > 0, "No artifacts found, please run generate_artifacts() first"

        if not isinstance(path, Path):
            path = Path(path)

        is_dir = len(path.suffix) == 0
        if is_dir:
            assert (
                path.is_dir()
            ), "The supplied path does not exists."  # Make sure it actually exists (we do not create it by default)
            for artifact in self.artifacts:
                extract = artifact.fmt == ArtifactFormat.MLF
                artifact.export(path, extract=extract)
                # TODO: move the following to a helper function and share code
                # dest = path / artifact.name
                # with open(dest, "w") as outfile:
                #     logger.info(f"Exporting artifact: {artifact.name}")
                #     outfile.write(artifact.content)
        else:
            assert path.parent.is_dir(), "The parent directory does not exist. Make sure to create if beforehand."
            # Warning: the first artifact is considered as main
            # We need to ensure that all further artifacts share the same prefix
            main_prefix = None
            for artifact in self.artifacts:
                if not main:
                    main_prefix = artifact.name
                else:
                    if main_prefix not in artifact.name:
                        logger.warn(
                            f"Skipping export of artifact '{artifact.name}' to prevent overwriting random files"
                        )
                        continue
                dest = path / artifact.name
                with open(dest, "w") as outfile:
                    logger.info(f"Exporting artifact: {artifact.name}")
                    outfile.write(artifact.content)

    def get_platform_config(self, platform):
        return {}

    def add_platform_config(self, platform, config):
        config.update(self.get_platform_config(platform))

    def get_platform_defs(self, platform):
        if platform == "espidf":
            backend_upper = self.name.upper()
            return {f"MLONMCU_BACKEND_{backend_upper}": True}
        else:
            return {"MLONMCU_BACKEND": self.name}

    def add_platform_defs(self, platform, defs):
        defs.update(self.get_platform_defs(platform))

    def supports_model(self, model):
        return any(fmt in self.supported_formats for fmt in model.formats)


def get_parser(backend_name, features, required, defaults):
    # TODO: add help strings should start with a lower case letter
    parser = argparse.ArgumentParser(
        description=f"Run {backend_name} backend",
        formatter_class=argparse.RawTextHelpFormatter,
        # epilog="""Use environment variables to overwrite default paths:""",
    )
    parser.add_argument("model", metavar="MODEL", type=str, nargs=1, help="Model to process")
    parser.add_argument(
        "--output",
        "-o",
        metavar="DIR",
        type=str,
        default=os.path.join(os.getcwd(), "out"),  # TODO: keep this or require flag?
        help="""Output directory/file (default: %(default)s)""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed messages for easier debugging (default: %(default)s)",
    )
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print the generated code to the command line instead (default: %(default)s)",  # TODO: instead or both?
    )
    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        metavar="FEATURE",
        # nargs=1,
        action="append",
        choices=list(dict.fromkeys(features)),
        help="Enabled features for the backend (default: %(default)s choices: %(choices)s)",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help="""Set a number of key-value pairs

Allowed options:
"""
        + "\n".join(
            [f"- [{backend_name}].{key} (Default: {value})" for key, value in defaults.items()]
            + [f"- {key} (required)" for key in required]
        ),
    )
    return parser


def init_backend_features(names, config):
    raise NotImplementedError
    # features = []
    # for name in names:
    #     feature_classes = get_supported_features(feature_type=FeatureType.BACKEND, feature_name=name)
    #     for feature_class in feature_classes:
    #         features.append(feature_class(config=config))
    # return features


def main(backend, args=None):
    parser = get_parser(
        backend.name,
        features=backend.FEATURES,
        required=backend.REQUIRED,
        defaults=backend.DEFAULTS,
    )
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    # TODO: handle args!
    model = Path(args.model[0])
    config = extract_config(args)
    features_names = extract_feature_names(args)
    features = init_backend_features(features_names, config)
    backend_inst = backend(features=features, config=config)
    backend_inst.load_model(model)
    if args.verbose:
        config["print_outputs"] = True
    backend_inst.generate_artifacts()
    if args.print:
        print("Printing generated artifacts:")
        for artifact in backend_inst.artifacts:
            print(f"=== {artifact.name} ===")
            artifact.print_summary()
            print("=== End ===")
            print()
    else:
        out = args.output
        backend_inst.export_code(out)
