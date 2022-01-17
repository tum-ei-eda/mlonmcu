import multiprocessing
from mlonmcu.target import SUPPORTED_TARGETS
from mlonmcu.feature.features import get_available_feature_names


def add_flow_options(parser):
    flow_parser = parser.add_argument_group("flow options")
    flow_parser.add_argument(
        "-t",
        "--target",
        type=str,
        metavar="TARGET",
        choices=SUPPORTED_TARGETS.keys(),
        default="etiss/pulpino",
        nargs=1,
        help="The target device/architecture (default: %(default)s choices: %(choices)s)",
    )
    flow_parser.add_argument(
        "-f",
        "--feature",
        type=str,
        metavar="FEATURE",
        # nargs=1,
        action="append",
        choices=get_available_feature_names(),
        help="Enabled features for target/framework/backend (default: %(default)s choices: %(choices)s)",
    )
    flow_parser.add_argument(
        "-c",
        "--config",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help="Set a number of key-value pairs "
        "(do not put spaces before or after the = sign). "
        "If a value contains spaces, you should define "
        "it with double quotes: "
        'foo="this is a sentence". Note that '
        "values are always treated as strings.",
    )
    flow_parser.add_argument(
        "--parallel",
        metavar="THREADS",
        nargs="?",
        type=int,
        const=multiprocessing.cpu_count(),
        default=1,
        help="Use multiple threads to process runs in parallel (%(const)s if specified, else %(default)s)",
    )
    flow_parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Display progress bar (default: %(default)s)",
    )
    flow_parser.add_argument(
        "--docker",
        action="store_true",
        help="Execute run.py inside docker container (default: %(default)s)",
    )


def add_common_options(parser):
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed messages for easier debugging (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce number of logging statements to a minimum (default: %(default)s)",
    )


def add_context_options(parser, with_home=True):
    common = parser.add_argument_group("context options")
    if with_home:
        common.add_argument(
            "-H",
            "--home",
            type=str,
            default=".",
            help="The path to the mlonmcu environment (overwriting $MLONMCU_HOME environment variable)",
        )


def add_model_options(parser):
    parser.add_argument(
        "models",
        metavar="model",
        type=str,
        nargs="*",
        default=None,
        help="Model to process",
    )
