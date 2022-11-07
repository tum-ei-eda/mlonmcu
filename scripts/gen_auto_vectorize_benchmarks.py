import os
import argparse
import multiprocessing
import logging

# import mlonmcu
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.session.run import RunStage
from mlonmcu.models.lookup import apply_modelgroups
from mlonmcu.logging import get_logger, set_log_level

logger = get_logger()

# MURISCVNN_TOOLCHAIN = "gcc"

FRONTEND = "tflite"

TARGETS = [
    "spike",
    "ovpsim",
    "etiss_pulpino",
    "riscv_qemu",
]

AUTOTUNED_TARGETS = [
    "spike",
    "ovpsim",
    "etiss_pulpino",
    # "riscv_qemu",
    # "corstone300",
]

DEFAULT_TARGETS = [
    # "spike",
    "ovpsim",
    # "etiss_pulpino",
    # "riscv_qemu",
    # "corstone300",
]

PLATFORM = "mlif"

TOOLCHAINS = ["gcc", "llvm"]

DEFAULT_TOOLCHAINS = ["gcc", "llvm"]

BACKENDS = ["tvmaot", "tflmi"]
DEFAULT_BACKENDS = ["tvmaot", "tflmi"]

VALIDATE_FEATURES = ["validate", "debug"]

BACKEND_DEFAULT_FEATURES = {"tvmaot": ["unpacked_api", "usmp"], "tflmi": []}


def get_backend_config(backend, features, enable_autotuned=False):
    BACKEND_FEATURES = {
        "tvmaot": [{"tvmaot.desired_layout": "NCHW"}, {"tvmaot.desired_layout": "NHWC"}],
        "tflmi": [{}],
    }
    ret = BACKEND_FEATURES[backend]
    return ret


DEFAULT_CONFIG = {
    "mlif.num_threads": 4,
}

BACKEND_DEFAULT_CONFIG = {
    "tflmi": {},
    "tvmaot": {"usmp.algorithm": "hill_climb"},  # Warning: usmp not enabled!
}

VLENS = [64, 128, 256, 512, 1024]

DEFAULT_VLENS = [128, 256, 512, 1024]

MODELS = [
    # "sine_model",
    # "magic_wand",
    # "micro_speech",
    # "cifar10",
    # "simple_mnist",
    "aww",
    "vww",
    "resnet",
    "toycar",
]

POSTPROCESSES = [
    "features2cols",
    "config2cols",
    "rename_cols",
    "filter_cols",
]

POSTPROCESS_CONFIG = {
    "filter_cols.keep": [
        "Model",
        "Backend",
        "Target",
        "Cycles",
        "Runtime [s]",
        "Total ROM",
        "Total RAM",
        # "ROM read-only",
        # "ROM code",
        # "ROM misc",
        # "RAM data",
        # "RAM zero-init data",
        "Incomplete",
        "Failing",
        "Features",
        # "Comment",
        "Validation",
        "Schedules",
        "Extensions",
        "VLEN",
        "Layout",
        "Toolchain",
        "Vectorize",
    ],
    "rename_cols.mapping": {
        "config_vext.vlen": "VLEN",
        "config_vext.elen": "ELEN",
        # "spike.vlen": "VLEN",
        # "etiss_pulpino.vlen": "VLEN",
        # "riscv_qemu.vlen": "VLEN",
        # "ovpsim.elen": "ELEN",
        # "spike.elen": "ELEN",
        # "etiss_pulpino.elen": "ELEN",
        # "riscv_qemu.elen": "ELEN",
        "config_tvmaot.desired_layout": "Layout",
        "config_mlif.toolchain": "Toolchain",
        "feature_auto_vectorize": "Vectorize",
    },
    "filter_cols.drop_nan": True,
}


def gen_features(backend, features, validate=False):
    ret = []
    ret.extend(BACKEND_DEFAULT_FEATURES[backend])
    if validate:
        ret += VALIDATE_FEATURES
    ret += features
    return ret


def gen_config(backend, backend_config, features, vlen, toolchain, enable_postprocesses=False):
    ret = {}
    ret.update(DEFAULT_CONFIG)
    ret.update(BACKEND_DEFAULT_CONFIG[backend])
    ret.update(backend_config)
    ret.update({"mlif.toolchain": toolchain})
    if enable_postprocesses:
        ret.update(POSTPROCESS_CONFIG)
    if "auto_vectorize" in features:
        for feature in features:
            if feature == "vext":
                ret["vext.vlen"] = vlen
    return ret


def benchmark(args):
    with MlonMcuContext() as context:
        user_config = context.environment.vars  # TODO: get rid of this workaround
        session = context.create_session()
        models = apply_modelgroups(args.models, context=context)
        for model in models:
            for backend in args.backend:
                for target in args.target:
                    for target_features in [[], ["vext", "auto_vectorize"]]:
                        for backend_config in get_backend_config(backend, target_features):
                            vlens = [0]
                            if "vext" in target_features:
                                vlens = args.vlen
                            features = gen_features(backend, target_features, validate=args.validate)
                            for vlen in vlens:
                                for toolchain in args.toolchain:
                                    if toolchain == "llvm" and "pext" in features:
                                        continue  # TODO: move this check up!
                                    config = gen_config(
                                        backend,
                                        backend_config,
                                        features,
                                        vlen,
                                        toolchain,
                                        enable_postprocesses=args.post,
                                    )
                                    config.update(user_config)  # TODO
                                    # resolve_missing_configs(config, features, target, context)
                                    # print("RUN", model, config, features, backend, target)
                                    run = session.create_run(config=config)
                                    run.add_features_by_name(features, context=context)
                                    run.add_platform_by_name(PLATFORM, context=context)
                                    run.add_frontend_by_name(FRONTEND, context=context)
                                    run.add_model_by_name(model, context=context)
                                    run.add_backend_by_name(backend, context=context)
                                    run.add_target_by_name(target, context=context)
                                    if args.post:
                                        run.add_postprocesses_by_name(POSTPROCESSES)
        if args.noop:
            stage = RunStage.LOAD
        else:
            stage = RunStage.RUN
        session.process_runs(until=stage, num_workers=args.parallel, progress=args.progress, context=context)
        report = session.get_reports()
        report_file = args.output
        report.export(report_file)
        print()
        print(report.df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        metavar="model",
        type=str,
        # nargs="+",
        nargs="*",
        default=MODELS,
        help="Model to process",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        action="append",
        choices=BACKENDS,
        # default=DEFAULT_BACKENDS,
        default=[],
        help=f"Backends to use (default: {DEFAULT_BACKENDS})",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        action="append",
        choices=TARGETS,
        # default=DEFAULT_TARGETS,
        default=[],
        help=f"Targets to use (default: {DEFAULT_TARGETS}s)",
    )
    parser.add_argument(
        "--vlen",
        type=int,
        action="append",
        choices=VLENS,
        # default=DEFAULT_VLENS,
        default=[],
        help=f"VLENS to use (RISC-V only) (default: {DEFAULT_VLENS})",
    )
    parser.add_argument(
        "--toolchain",
        type=str,
        action="append",
        choices=TOOLCHAINS,
        # default=DEFAULT_TOOLCHAINS,
        default=[],
        help=f"Toolchain to use (default: {DEFAULT_TOOLCHAINS})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate model outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--autotuned",
        action="store_true",
        help="Use tuning records, if available (default: %(default)s)",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Run postprocesses after the session (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Display progress bar (default: %(default)s)",
    )
    parser.add_argument(
        "--parallel",
        metavar="THREADS",
        nargs="?",
        type=int,
        const=multiprocessing.cpu_count(),
        default=1,
        help="Use multiple threads to process runs in parallel (%(const)s if specified, else %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        type=str,
        default=os.path.join(os.getcwd(), "out.csv"),
        help="""Output CSV file (default: %(default)s)""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed messages for easier debugging (default: %(default)s)",
    )
    parser.add_argument(
        "--noop",
        action="store_true",
        help="Skip processing runs. (default: %(default)s)",
    )
    args = parser.parse_args()
    if not args.backend:
        args.backend = DEFAULT_BACKENDS
    if not args.target:
        args.target = DEFAULT_TARGETS
    if not args.vlen:
        args.vlen = DEFAULT_VLENS
    if not args.toolchain:
        args.toolchain = DEFAULT_TOOLCHAINS
    if args.verbose:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.INFO)
    benchmark(args)


if __name__ == "__main__":
    main()
