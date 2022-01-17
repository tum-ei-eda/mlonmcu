import os
import sys
import logging
import argparse
import tempfile
from contextlib import closing
from pathlib import Path
from typing import List

from mlonmcu.context import MlonMcuContext
from mlonmcu.setup import utils  # TODO: Move one level up?
from mlonmcu.cli.helper.parse import extract_features, extract_config
from mlonmcu.flow import SUPPORTED_BACKENDS, SUPPORTED_FRAMEWORKS
from mlonmcu.target import SUPPORTED_TARGETS
from .inout import write_inout_data

logger = logging.getLogger("mlonmcu")


class MLIF:
    """Model Library Interface class."""

    FEATURES = ["debug"]
    DEFAULTS = {"debug": False}
    # REQUIRES = ["mlif.src_dir"]
    REQUIRES = []

    def __init__(
        self, framework, backend, target, features=None, config=None, context=None
    ):
        self.framework = framework  # TODO: required? or self.target.framework?
        self.backend = backend
        self.target = target
        self.features = features if features else []
        self.process_features()
        self.config = (
            config if config else {}
        )  # Warning: this should only be used for passing paths,.. when no co context is provided, NO customization of the build is allowed!
        self.filter_config()
        self.ignore_data = False
        if (
            "mlif.ignore_data" in self.config
        ):  # TODO: rather introduce CompileFeature(inout)?
            self.ignore_data = bool(self.config["mlif.ignore_data"])
        self.context = context
        self.goal = "generic_mlif"
        flags = [self.framework.shortname, self.backend.shortname, self.target.name] + [
            feature.name for feature in self.features
        ]
        dir_name = utils.makeDirName("mlif", flags=flags)
        self.tempdir = None
        if "mlif.build_dir" in self.config:
            if self.context:
                logger.warn("User has overwritten the value of 'mlif.build_dir'")
            self.build_dir = Path(self.config["mlif.build_dir"])
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
        self.data_dir = self.build_dir / "data"
        if "mlif.src_dir" in self.config:
            if self.context:
                logger.warn("User has overwritten the value of 'mlif.src_dir'")
            self.mlif_dir = Path(self.config["mlif.src_dir"])
        else:
            if context:
                assert "sw" in context.environment.paths
                self.mlif_dir = context.environment.paths[
                    "sw"
                ]  # TODO: clone on environment init!
            else:
                raise RuntimeError(
                    "Please define the value of 'mlif.src_dir' or pass a context"
                )
        self.validate()

    def process_features(self):
        for feature in self.features:
            if FeatureType.COMPILE in feature.types:
                assert (
                    feature.name in self.FEATURES
                ), f"Incompatible compile feature:   {feature.name}"
                feature.add_compile_config(self.config)

    def remove_config_prefix(self, config):
        def helper(key):
            return key.split(f"{self.name}.")[-1]

        return {helper(key): value for key, value in config if f"{self.name}." in key}

    def filter_config(self):
        cfg = self.remove_config_prefix(self.config)
        for required in self.REQUIRED:
            value = None
            if required in cfg:
                value = cfg[required]
            elif required in self.config:
                value = self.config[required]
            assert value is not None, f"Required config key can not be None: {required}"

        for key in self.DEFAULTS:
            if key not in cfg:
                cfg[key] = self.DEFAULTS[key]

        for key in cfg:
            if key not in self.DEFAULTS.keys() + self.REQUIRED:
                logger.warn("Backend received an unknown config key: %s", key)
                del cfg[key]

        self.config = cfg

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def validate(self):
        print("self.framework", self.framework.shortname)
        assert self.framework.shortname in SUPPORTED_FRAMEWORKS
        # assert self.framework.shortname in enabled_frameworks  # TODO
        assert self.backend.shortname in SUPPORTED_BACKENDS
        # assert self.backend.shortname in enabled_backends  # TODO
        for feature in self.features:
            # TODO: maybe rather for type in types?
            if feature.type == FeatureType.FRAMEWORK:
                assert feature in self.framework.features
            if feature.type == FeatureType.BACKEND:
                assert feature in self.backend.features
            if feature.type == FeatureType.TARGET:
                assert feature in self.target.features

    def get_common_cmake_args(self, model, num=1):
        args = {}
        return args

    def prepare(self, model, ignore_data=False):
        utils.mkdirs(self.data_dir)
        data_file = self.data_dir / "data.c"
        write_inout_data(model, data_file, skip=ignore_data)
        return data_file

    def configure(self, src, model, num=1, debug=False):
        if not isinstance(src, Path):
            src = Path(src)
        cmakeArgs = []
        cmakeArgs.extend(self.framework.get_cmake_args())
        cmakeArgs.extend(self.backend.get_cmake_args())
        cmakeArgs.extend(self.target.get_cmake_args())
        cmakeArgs.extend(self.get_common_cmake_args(model))
        if src.is_file():
            src = src.parent  # TODO deal with directories or files?
        if src.is_dir():
            cmakeArgs.append("-DSRC_DIR=" + str(src))
        else:
            raise RuntimeError("Unable to find sources!")
        data_file = self.prepare(model, ignore_data=(not debug or self.ignore_data))
        cmakeArgs.append("-DDATA_SRC=" + str(data_file))
        utils.mkdirs(self.build_dir)
        utils.cmake(self.mlif_dir, *cmakeArgs, cwd=self.build_dir, debug=debug)

    def compile(self, src=None, model=None, num=1, debug=True):
        if src:
            self.configure(src, model, num=num, debug=debug)
        utils.make(self.goal, cwd=self.build_dir)

    def export(self):
        raise NotImplementedError


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
    features = extract_features(args)
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
