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
    kickoff_runs,
)
from mlonmcu.config import resolve_required_config
from mlonmcu.cli.load import handle as handle_load, add_load_options, add_model_options
from mlonmcu.flow import SUPPORTED_BACKENDS, SUPPORTED_FRAMEWORKS
from mlonmcu.session.run import RunStage


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
    add_load_options(parser)
    add_flow_options(parser)
    return parser


def _handle(context, args):
    handle_load(args, ctx=context)
    backend_names = args.backend
    if isinstance(backend_names, list) and len(backend_names) > 0:
        backends = backend_names
    elif isinstance(backend_names, str):
        backends = [backend_names]
    else:
        assert backend_names is None, "TODO"
        frameworks = context.environment.get_default_frameworks()
        backends = []
        for framework in frameworks:
            framework_backends = context.environment.get_default_backends(framework)
            backends.extend(framework_backends)
    assert len(context.sessions) > 0
    session = context.sessions[-1]
    new_runs = []
    for run in session.runs:
        for backend_name in backends:
            new_run = run.copy()
            # TODO: where to add framework features/config?
            backend_cls = SUPPORTED_BACKENDS[backend_name]
            required_keys = backend_cls.REQUIRED
            backend_config = new_run.config.copy()
            backend_config.update(
                resolve_required_config(
                    backend_cls.REQUIRED,
                    features=new_run.features,
                    config=new_run.config,
                    cache=context.cache,
                )
            )
            backend = backend_cls(features=new_run.features, config=backend_config)
            framework_name = backend.framework
            framework_cls = SUPPORTED_FRAMEWORKS[framework_name]
            required_keys = backend_cls.REQUIRED
            framework_config = new_run.config.copy()
            framework_config.update(
                resolve_required_config(
                    framework_cls.REQUIRED,
                    features=new_run.features,
                    config=new_run.config,
                    cache=context.cache,
                )
            )
            framework = framework_cls(
                features=new_run.features, config=framework_config
            )
            new_run.add_backend(backend)
            new_run.add_framework(framework)
            new_runs.append(new_run)

    session.runs = new_runs


def handle(args, ctx=None):
    if ctx:
        _handle(ctx, args)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle(context, args)
            kickoff_runs(args, RunStage.BUILD, context)
