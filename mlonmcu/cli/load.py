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
from mlonmcu.feature.features import get_available_features
from .helper.parse import extract_feature_names, extract_config


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
    # TODO: move to helper function OUT OF CLI
    feature_names = extract_feature_names(args)
    config = extract_config(args)
    features = []
    for feature_name in feature_names:
        available_features = get_available_features(feature_name=feature_name)
        for feature_cls in available_features:
            feature_inst = feature_cls(config=config)
            features.append(feature_inst)

        # How about featuretype.other?

    session = context.get_session()
    for model_name in model_names:
        path = None  # FIXME
        model = Model(model_name, path)
        run = Run(model=model, features=features, config=config)
        session.runs.append(run)
    for run in session.runs:
        loaded_model = load_model(run.model, context=context)
        run.artifacts["model"] = loaded_model


def handle(args, ctx=None):
    if ctx:
        _handle(ctx, args)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle(context, args)
