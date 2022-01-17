"""Command line subcommand for the build process."""

import copy

import mlonmcu
from mlonmcu.flow import get_available_backend_names
import mlonmcu.flow.tflite
import mlonmcu.flow.tvm
from mlonmcu.models.model import Model
from mlonmcu.session.run import Run
from mlonmcu.session.session import Session
from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
    add_model_options,
    add_flow_options,
)
from mlonmcu.cli.load import handle as handle_load, add_load_options, add_model_options
from .helper.parse import extract_feature_names, extract_config


def add_build_options(parser):
    # TODO: rename to build_group
    build_parser = parser.add_argument_group("build options")
    build_parser.add_argument(
        "-b",
        "--backend",
        type=str,
        action="append",
        choices=get_available_backend_names(),
        help="Backends to use (default: %(default)s)",
    )


def get_parser(subparsers, parent=None):
    """ "Define and return a subparser for the build subcommand."""
    parser = subparsers.add_parser(
        "build",
        description="Build model using the ML on MCU flow.",
        parents=[parent] if parent else [],
        add_help=(parent is None),
    )
    parser.set_defaults(func=handle)
    add_model_options(parser)
    add_common_options(parser)
    add_context_options(parser)
    add_build_options(parser)
    add_flow_options(parser)
    return parser


def _handle(context, args):
    handle_load(args, ctx=context)
    print(args)
    configs = extract_config(args)
    print("configs", configs)
    # print(configs)
    # input()
    backends_names = args.backend
    assert len(context.sessions) > 0
    session = context.sessions[-1]
    print("session", session)
    print("backends_names", backends_names)
    new_runs = []
    for run in session.runs:
        if backends_names and len(backends_names) > 0:
            for backend_name in backends_names:
                new_run = copy.deepcopy(run)
                backend_class = SUPPORTED_BACKENDS[backend_name]
                backend = backend_class(config=configs)
                new_run.backend = backend
                new_run.cfg = configs
                new_runs.append(new_run)
        else:
            raise NotImplementedError("TODO: Default backends!")
    session.runs = new_runs
    for run in session.runs:
        run.build(context=context)
    print("session.runs", session.runs)


def handle(args, ctx=None):
    print("HANDLE BUILD")
    if ctx:
        _handle(ctx, args)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle(context, args)
    print("HANDLED BUILD")
