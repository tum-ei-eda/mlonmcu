"""Command line subcommand for the load stage."""

import copy

import mlonmcu
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
from .helper.parse import extract_config_and_init_features


def add_load_options(parser):
    load_parser = parser.add_argument_group("load options")


def get_parser(subparsers):
    """ "Define and return a subparser for the load subcommand."""
    parser = subparsers.add_parser(
        "load", description="Load model using the ML on MCU flow."
    )
    parser.set_defaults(func=handle)
    add_model_options(parser)
    add_common_options(parser)
    add_context_options(parser)
    add_load_options(parser)
    add_flow_options(parser)
    return parser


def load_model(model, context=None):
    pass


def _handle(context, args):
    model_names = args.models
    config, features = extract_config_and_init_features(args)

    session = context.get_session()
    for model_name in model_names:
        path = None  # FIXME
        # model = Model(model_name, path)
        run = Run(model_name=model_name, features=features, config=config)
        session.runs.append(run)
    for run in session.runs:
        run.load(context=context)


def handle(args, ctx=None):
    if ctx:
        _handle(ctx, args)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle(context, args)
