import os
import sys
import logging
import argparse
import tempfile
from contextlib import closing
from pathlib import Path
from typing import List

from mlonmcu.setup import utils  # TODO: Move one level up?
from mlonmcu.cli.helper.parse import extract_feature_names, extract_config
from mlonmcu.flow import SUPPORTED_BACKENDS, SUPPORTED_FRAMEWORKS
from mlonmcu.target import SUPPORTED_TARGETS
from mlonmcu.config import filter_config
from mlonmcu.feature.features import get_matching_features
from mlonmcu.feature.type import FeatureType
from mlonmcu.artifact import Artifact, ArtifactFormat

from mlonmcu.logging import get_logger

logger = get_logger()


class MLIF:
    """Model Library Interface class."""

    FEATURES = ["debug", "validate"]

    DEFAULTS = {"debug": False, "ignore_data": True, "build_dir": None}

    # REQUIRES = ["mlif.src_dir"]
    REQUIRED = []

    def __init__(
        self, framework, backend, target, features=None, config=None, context=None
    ):
        self.framework = framework  # TODO: required? or self.target.framework?
        self.backend = backend
        self.target = target
        self.config = (
            config if config else {}
        )  # Warning: this should only be used for passing paths,.. when no co context is provided, NO customization of the build is allowed!
        self.features = self.process_features(features)
        self.config = filter_config(self.config, "mlif", self.DEFAULTS, self.REQUIRED)
        self.context = context
        self.goal = "generic_mlif"
        flags = [self.framework.name, self.backend.name, self.target.name] + [
            feature.name for feature in self.features
        ]
        dir_name = utils.makeDirName("mlif", flags=flags)
        self.tempdir = None
        if self.config["build_dir"]:
            # if self.context:
            #     logger.warn("User has overwritten the value of 'mlif.build_dir'")
            self.build_dir = Path(self.config["build_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.build_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.info(
                    "Creating temporary directory because no context was available and 'mlif.build_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.build_dir = Path(self.tempdir.name) / dir_name
                logger.info("Temporary build directory: %s", self.build_dir)
        # self.data_dir = self.build_dir / "data"
        if "mlif.src_dir" in self.config:
            if self.context:
                logger.warn("User has overwritten the value of 'mlif.src_dir'")
            self.mlif_dir = Path(self.config["mlif.src_dir"])
        else:
            if context:
                # assert "sw" in context.environment.paths
                # self.mlif_dir = context.environment.paths[
                #     "sw"
                # ]  # TODO: clone on environment init!
                self.mlif_dir = (
                    Path(context.environment.home) / "sw"
                )  # TODO: Define in env paths
            else:
                raise RuntimeError(
                    "Please define the value of 'mlif.src_dir' or pass a context"
                )
        self.validate()

    @property
    def debug(self):
        return bool(self.config["debug"])

    @property
    def ignore_data(self):
        return bool(self.config["ignore_data"])

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.COMPILE)
        for feature in features:
            assert (
                feature.name in self.FEATURES
            ), f"Incompatible feature: {feature.name}"
            feature.add_compile_config(self.config)
        return features

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def validate(self):
        pass
        # assert self.framework.name in SUPPORTED_FRAMEWORKS
        # # assert self.framework.name in enabled_frameworks  # TODO
        # assert self.backend.name in SUPPORTED_BACKENDS
        # # assert self.backend.name in enabled_backends  # TODO
        # for feature in self.features:
        #     # TODO: maybe rather for type in types?
        #     if feature.type == FeatureType.FRAMEWORK:
        #         assert feature in self.framework.features
        #     if feature.type == FeatureType.BACKEND:
        #         assert feature in self.backend.features
        #     if feature.type == FeatureType.TARGET:
        #         assert feature in self.target.features

    # def get_common_cmake_args(self, model, num=1):
    def get_common_cmake_args(self, num=1):
        args = []
        args.append(f"-DNUM_RUNS={num}")
        return args

    # def prepare(self, model, ignore_data=False):
    def prepare(self):
        # utils.mkdirs(self.data_dir)
        # data_file = self.data_dir / "data.c"
        # write_inout_data(model, data_file, skip=ignore_data)
        # return data_file
        pass

    def configure(self, src, model, num=1, data_file=None):
        if not isinstance(src, Path):
            src = Path(src)
        cmakeArgs = []
        cmakeArgs.extend(self.framework.get_cmake_args())
        cmakeArgs.extend(self.backend.get_cmake_args())
        cmakeArgs.extend(self.target.get_cmake_args())
        cmakeArgs.extend(self.get_common_cmake_args(num=num))
        if src.is_file():
            src = src.parent  # TODO deal with directories or files?
        if src.is_dir():
            cmakeArgs.append("-DSRC_DIR=" + str(src))
        else:
            raise RuntimeError("Unable to find sources!")
        # data_file = self.prepare(model, ignore_data=(not debug or self.ignore_data))
        if self.ignore_data:
            cmakeArgs.append("-DDATA_SRC=")
        else:
            assert data_file is not None, "No data.c file was supplied"
            cmakeArgs.append("-DDATA_SRC=" + str(data_file))
        utils.mkdirs(self.build_dir)
        utils.cmake(self.mlif_dir, *cmakeArgs, cwd=self.build_dir, debug=self.debug)

    def compile(self, src=None, model=None, num=1, data_file=None):
        if src:
            self.configure(src, model, num=num, data_file=data_file)
        utils.make(self.goal, cwd=self.build_dir)

    def generate_elf(self, src=None, model=None, num=1, data_file=None):
        artifacts = []
        self.compile(src=src, model=model, num=num, data_file=data_file)
        elf_file = self.build_dir / "bin" / "generic_mlif"
        # TODO: just use path instead of raw data?
        with open(elf_file, "rb") as handle:
            data = handle.read()
            artifact = Artifact("generic_mlif", raw=data, fmt=ArtifactFormat.RAW)
            artifacts.append(artifact)
        self.artifacts = artifacts

    def export_elf(self, path):
        assert (
            len(self.artifacts) > 0
        ), "No artifacts found, please run generate_elf() first"

        if not isinstance(path, Path):
            path = Path(path)
        assert (
            path.is_dir()
        ), "The supplied path does not exists."  # Make sure it actually exists (we do not create it by default)
        for artifact in self.artifacts:
            artifact.export(path)


def add_common_options(parser: argparse.ArgumentParser):
    """Add a set of common options to a command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The command line parser
    """
    common_group = parser.add_argument_group("MLIF options")
    common_group.add_argument(
        "source",
        metavar="SRC",
        type=str,
        help="The generated input source files (or a directory)",
    )
    common_group.add_argument(
        "-b",
        "--backend",
        metavar="BACKEND",
        default=None,
        help="The used backend (required)",
    )  # TODO
    common_group.add_argument(
        "-t",
        "--target",
        metavar="TARGET",
        default=None,
        help="The target to use (required)",
    )  # TODO
    common_group.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        default=None,
        help="The model diretory for mlif support files (optional)",
    )  # TODO
    common_group.add_argument(
        "-o",
        "--out-dir",
        dest="out",
        metavar="DIR",
        default=None,
        help="The build directory (optional)",
    )  # TODO
    common_group.add_argument(
        "-f",
        "--feature",
        type=str,
        metavar="FEATURE",
        action="append",
        # choices=list(dict.fromkeys(ALL_FEATURES)), # TODO: get from selected target?
        help="Enabled features for target",
    )
    common_group.add_argument(
        "-c",
        "--config",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help="Custom target config as key-value pairs",
    )


def add_configure_options(parser: argparse.ArgumentParser):
    """Add a set of options to a command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The command line parser
    """
    configure_group = parser.add_argument_group("Configure options")
    configure_group.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="If the project should be initialized in debug mode.",
    )


# TODO: reuse somewhere else?
def get_backend_by_name(name, features=None, config=None):
    assert name in SUPPORTED_BACKENDS.keys(), f"Unknown backend name: {name}"
    backend = SUPPORTED_BACKENDS[name](features=None, config=None)
    return backend


def get_framework_by_name(name, features=None, config=None):
    assert name in SUPPORTED_FRAMEWORKS.keys(), f"Unkown framework name: {name}"
    framework = SUPPORTED_FRAMEWORKS[name](features=features, config=config)
    return framework


def get_target_by_name(name, features=None, config=None):
    assert name in SUPPORTED_TARGETS.keys(), f"Unknown target name: {name}"
    target = SUPPORTED_TARGETS[name](features=features, config=config)
    return target


def get_model(model):
    return None


def create_mlif_from_args(args):
    features = extract_feature_names(args)
    config = extract_config(args)
    assert args.backend is not None, "The backend must be set"
    backend = get_backend_by_name(args.backend, features=features, config=config)
    assert args.backend is not None, "The target must be set"
    target = get_target_by_name(args.target, features=features, config=config)
    framework = get_framework_by_name(
        backend.framework, features=features, config=config
    )
    if args.out:
        assert (
            "mlif.build_dir" not in config
        ), "The value of 'mlif.build_dir' should be set with the --out-dir argument"
        config["mlif.build_dir"] = Path(args.out)
    return MLIF(framework, backend, target, features=features, config=config)


def main(args=None):
    parser = argparse.ArgumentParser(description="ML on MCU MLIF")
    subparsers = parser.add_subparsers(dest="subcommand")
    configure_parser = subparsers.add_parser("configure", description="Configure MLIF")

    def _handle_configure(args):
        with closing(create_mlif_from_args(args)) as mlif_inst:
            source = args.source
            debug = args.debug
            model = get_model(args.model)  # TODO: only get modeldata?
            mlif_inst.configure(source, model=model, debug=debug)

    configure_parser.set_defaults(func=_handle_configure)
    add_common_options(configure_parser)
    add_configure_options(configure_parser)

    compile_parser = subparsers.add_parser(
        "compile", description="Inspect program with target"
    )

    def _handle_compile(args):
        with closing(create_mlif_from_args(args)) as mlif_inst:
            source = args.source
            debug = args.debug
            model = get_model(args.model)  # TODO: only get modeldata?
            mlif_inst.compile(src=source, model=model, debug=debug)

    compile_parser.set_defaults(func=_handle_compile)
    add_common_options(compile_parser)
    add_configure_options(compile_parser)
    compile_group = compile_parser.add_argument_group("Compile options")
    compile_group.add_argument(
        "-e",
        "--out-elf",
        metavar="ELF",
        default=None,
        const="./generic_mlif",
        nargs="?",
        help="The build directory (optional)",
    )
    # TODO: allow to specify elf_destination (--out-elf)
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        raise RuntimeError("Invalid command. For usage details use '--help'!")


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))  # pragma: no cover
