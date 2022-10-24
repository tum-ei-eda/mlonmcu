import os
import argparse
import multiprocessing
import logging

# import mlonmcu
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.session.run import RunStage
from mlonmcu.session.postprocess.postprocess import SessionPostprocess
from mlonmcu.models.lookup import apply_modelgroups
from mlonmcu.logging import get_logger, set_log_level

logger = get_logger()

# MURISCVNN_TOOLCHAIN = "gcc"


class CustomPostprocess(SessionPostprocess):  # RunPostprocess?
    """TODO"""

    def __init__(self, features=None, config=None):
        super().__init__("custom", features=features, config=config)

    def post_session(self, report):
        """Called at the end of a session."""
        df = report.post_df.copy()
        df["Schedules"] = df.apply(
            lambda row: "ARM (Tuned)"
            if row.get("config_tvmaot.target_device", "") == "arm_cpu" and row.get("feature_autotuned")
            else "ARM"
            if row.get("config_tvmaot.target_device", "") == "arm_cpu"
            else "RISC-V (Tuned)"
            if row.get("feature_target_optimized") and row.get("feature_autotuned")
            else "RISC-V"
            if row.get("feature_target_optimized")
            else ("Default (Tuned)" if row.get("feature_autotuned") else "Default"),
            axis=1,
        )
        # TODO: allow combinations
        df["Extensions"] = df.apply(
            lambda row: "VEXT+PEXT"
            if row.get("feature_vext") and row.get("feature_pext")
            else ("VEXT" if row.get("feature_vext") else ("PEXT" if row.get("feature_pext") else (None))),
            axis=1,
        )
        report.post_df = df


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
]

DEFAULT_TARGETS = [
    # "spike",
    "ovpsim",
    # "etiss_pulpino",
    # "riscv_qemu",
]

PLATFORM = "mlif"

TOOLCHAINS = ["gcc", "llvm"]

DEFAULT_TOOLCHAINS = ["gcc", "llvm"]

BACKENDS = ["tvmaot"]
DEFAULT_BACKENDS = ["tvmaot"]

FEATURES = [
    "target_optimized",
    "auto_vectorize",
    "disable_legalize",
    "arm_schedules",
    "vext",
    "pext",
    "none",
]

DEFAULT_FEATURES = [
    "target_optimized",
    "auto_vectorize",
    "vext",
    "pext",
    # "disable_legalize",
]

TUNING_RECORDS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "resources",
    "frameworks",
    "tvm",
    "tuning_records",
    "riscv_cpu_v0.01.log.best",
)


def get_target_features(
    target,
    enable_default=False,
    enable_target_optimized=False,
    enable_auto_vectorize=False,
    enable_vext=False,
    enable_pext=False,
):
    TARGET_FEATURES = {
        "spike": [
            *(
                [
                    [],
                    *(
                        ([["vext", "auto_vectorize"]] if enable_auto_vectorize and enable_vext else [["vext"]])
                        if enable_vext
                        else []
                    ),
                    *([["pext"]] if enable_pext else []),
                ]
                if enable_default
                else []
            ),
            *(
                [
                    ["target_optimized"],
                    *(
                        (
                            [["target_optimized", "vext", "auto_vectorize"]]
                            if enable_auto_vectorize and enable_vext
                            else [["target_optimized", "vext"]]
                        )
                        if enable_vext
                        else []
                    ),
                    *([["target_optimized", "pext"]] if enable_pext else []),
                ]
                if enable_target_optimized
                else []
            ),
        ],
        "ovpsim": [
            *(
                [
                    [],
                    *(
                        ([["vext", "auto_vectorize"]] if enable_auto_vectorize and enable_vext else [["vext"]])
                        if enable_vext
                        else []
                    ),
                    *([["pext"]] if enable_pext else []),
                ]
                if enable_default
                else []
            ),
            *(
                [
                    ["target_optimized"],
                    *(
                        (
                            [["target_optimized", "vext", "auto_vectorize"]]
                            if enable_auto_vectorize and enable_vext
                            else [["target_optimized", "vext"]]
                        )
                        if enable_vext
                        else []
                    ),
                    *([["target_optimized", "pext"]] if enable_pext else []),
                ]
                if enable_target_optimized
                else []
            ),
        ],
        "riscv_qemu": [
            *(
                [
                    [],
                    *(
                        ([["vext", "auto_vectorize"]] if enable_auto_vectorize and enable_vext else [["vext"]])
                        if enable_vext
                        else []
                    ),
                ]
                if enable_default
                else []
            ),
            *(
                [
                    ["target_optimized"],
                    *(
                        (
                            [["target_optimized", "vext", "auto_vectorize"]]
                            if enable_auto_vectorize and enable_vext
                            else [["target_optimized", "vext"]]
                        )
                        if enable_vext
                        else []
                    ),
                ]
                if enable_target_optimized
                else []
            ),
        ],
    }
    return TARGET_FEATURES[target]


VALIDATE_FEATURES = ["validate", "debug"]

BACKEND_DEFAULT_FEATURES = {
    # "tvmaot": ["unpacked_api", "usmp"],
    "tvmaot": ["unpacked_api"],
}


def get_backend_features(backend, target, enable_autotuned=False, enable_disable_legalize=False):
    # print("get_backend_features", backend, target, enable_autotuned, enable_disable_legalize)
    BACKEND_FEATURES = {
        "tvmaot": [
            [],
            *([["autotuned"]] if enable_autotuned and target in AUTOTUNED_TARGETS else []),
            *([["disable_legalize"]] if enable_disable_legalize else []),
        ],
    }
    return BACKEND_FEATURES[backend]


def get_backend_config(backend, features, enable_autotuned=False, enable_arm_schedules=False):
    # print("get_backend_config", backend, features, enable_autotuned)
    BACKEND_FEATURES = {
        "tvmaot": [
            *(
                [
                    {"tvmaot.desired_layout": "NCHW"},
                    {"tvmaot.desired_layout": "NHWC"},
                    *(
                        [
                            {"tvmaot.desired_layout": "NCHW", "tvmaot.target_device": "arm_cpu"},
                            {"tvmaot.desired_layout": "NHWC", "tvmaot.target_device": "arm_cpu"},
                        ]
                        if enable_arm_schedules
                        else []
                    ),
                ]
                if "target_optimized" not in features
                else [{}]
            ),
        ],
    }
    ret = BACKEND_FEATURES[backend]
    if enable_autotuned and backend == "tvmaot":
        for cfg in ret:
            cfg.update({"autotuned.results_file": TUNING_RECORDS})
    # print("ret", ret)
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

POSTPROCESSES_0 = [
    "features2cols",
    "config2cols",
]

POSTPROCESSES_1 = [
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
        "ELEN",
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
    if "target_optimized" in features or "auto_vectorize" in features:
        for feature in features:
            if feature == "pext":
                assert vlen == 0
            elif feature == "vext":
                ret["vext.vlen"] = vlen
    return ret


def benchmark(args):
    # print("args", args)
    with MlonMcuContext() as context:
        user_config = context.environment.vars  # TODO: get rid of this workaround
        session = context.create_session()
        models = apply_modelgroups(args.models, context=context)
        for model in models:
            # print("model", model)
            for backend in args.backend:
                # print("backend", backend)
                for target in args.target:
                    # print("target", target)
                    enable_default = not args.skip_default
                    enable_target_optimized = "target_optimized" in args.feature
                    enable_auto_vectorize = "auto_vectorize" in args.feature
                    enable_vext = "vext" in args.feature
                    enable_pext = "pext" in args.feature
                    enable_arm_schedules = "arm_schedules" in args.feature
                    for target_features in get_target_features(
                        target,
                        enable_default=enable_default,
                        enable_target_optimized=enable_target_optimized,
                        enable_auto_vectorize=enable_auto_vectorize,
                        enable_vext=enable_vext,
                        enable_pext=enable_pext,
                    ):
                        # print("target_features", target_features)
                        enable_autotuned = False
                        if args.autotuned:
                            # if "target_optimized" not in target_features and backend == "tvmaot":
                            if backend == "tvmaot":
                                pass
                                # enable_autotuned = True
                        enable_disable_legalize = (
                            "disable_legalize" in args.feature and "target_optimized" not in target_features
                        )
                        for backend_features in get_backend_features(
                            backend,
                            target,
                            enable_autotuned=enable_autotuned,
                            enable_disable_legalize=enable_disable_legalize,
                        ):
                            # print("backend_features", backend_features)
                            features = list(set(target_features + backend_features))
                            # print("features", features)
                            for backend_config in get_backend_config(
                                backend,
                                features,
                                enable_autotuned=enable_autotuned,
                                enable_arm_schedules=enable_arm_schedules,
                            ):
                                vlens = [0]
                                if "vext" in features:
                                    if "target_optimized" in features or "auto_vectorize" in features:
                                        vlens = args.vlen
                                    else:
                                        continue
                                features = gen_features(backend, features, validate=args.validate)
                                for vlen in vlens:
                                    # print("vlen", vlen)
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
                                            run.add_postprocesses_by_name(POSTPROCESSES_0)
                                            run.add_postprocess(CustomPostprocess(), append=True)
                                            run.add_postprocesses_by_name(POSTPROCESSES_1, append=True)
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
        "-f",
        "--feature",
        type=str,
        action="append",
        choices=FEATURES,
        # default=default_features,
        default=[],
        help=f"features to use (default: {DEFAULT_FEATURES})",
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
        "--skip-default",
        dest="skip_default",
        action="store_true",
        help="Do not generate benchmarks for reference runs (default: %(default)s)",
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
    if not args.feature:
        args.feature = DEFAULT_FEATURES
    if not args.vlen:
        args.vlen = DEFAULT_VLENS
    if not args.toolchain:
        args.toolchain = DEFAULT_TOOLCHAINS
    if "none" in args.feature:
        assert len(args.feature) == 1
        args.feature = []
    if args.verbose:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.INFO)
    benchmark(args)


if __name__ == "__main__":
    main()
