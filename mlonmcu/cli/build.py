"""Command line subcommand for the build process."""

import copy

import mlonmcu
import mlonmcu.flow.tflite
import mlonmcu.flow.tvm
from mlonmcu.session.run import Run
from mlonmcu.session.session import Session

def add_build_options(parser):
    build_parser = parser.add_argument_group("build options")
    build_parser.add_argument(
        "-b",
        "--backend",
        type=str,
        action='append',
        choices=["tflmc", "tflmi", "tvmaot", "tvmrt", "tvmcg", "tflm", "utvm"],
        help="Backends to use (default: %(default)s)")

def get_build_parser(subparsers):
    """"Define and return a subparser for the build subcommand."""
    parser = subparsers.add_parser('build', description='Build model using the ML on MCU flow.')
    parser.set_defaults(func=handle)
    parser.add_argument(
        "models", metavar="model", type=str, nargs="*", default=None, help="Model to process"
    )
    add_build_options(parser)
    return parser



def load_model(model, context=None):
    pass

def _handle_load(context, args):
    models = args.models
    print("models", models)
    if len(context.sessions) == 0:
        session = Session()
        context.sessions.append(session)
    else:
        session = context.sessions[-1]
    for model in models:
        run = Run(model=model)
        session.runs.append(run)
    for run in session.runs:
        loaded_model = load_model(run.model, context=context)
        run.artifacts["model"] = loaded_model


def handle_load(args, ctx=None):
    print("HANLDE LOAD")
    if ctx:
        _handle_load(ctx, args)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle_load(context, args)
    print("HANLDED LOAD")

backend_classes = {
    'tflmc': mlonmcu.flow.tflite.TFLMCBackend,
    'tflmi': mlonmcu.flow.tflite.TFLMIBackend,
    'tvmaot': mlonmcu.flow.tvm.TVMAOTBackend,
    'tvmrt': mlonmcu.flow.tvm.TVMRTBackend,
    'tvmcg': mlonmcu.flow.tvm.TVMCGBackend,
}

def run_backend(run, context=None):
    print("CODEGEN RUN", run)
    backend_class = backend_classes[run.backend]
    backend = backend_class(config={}, context=context)
    backend.load(model=run.artifacts["model"])
    code = backend.generate_code()
    run.artifacts["code"] = code

def _handle(context, args):
    handle_load(args, ctx=context)
    backends = args.backend
    assert len(context.sessions) > 0
    session = context.sessions[-1]
    print("backends", backends)
    new_runs = []
    for run in session.runs:
        if backends and len(backends) > 0:
            for backend in backends:
                new_run = copy.deepcopy(run)
                new_run.backend = backend
                new_runs.append(new_run)
        else:
            raise NotImplementedError("TODO: Default backends!")
    session.runs = new_runs
    for run in session.runs:
        run_backend(run, context=context)
    print("session.runs", session.runs)

def handle(args, ctx=None):
    print("HANDLE BUILD")
    if ctx:
        _handle(ctx, args)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle(context, args)
    print("HANDLED BUILD")
