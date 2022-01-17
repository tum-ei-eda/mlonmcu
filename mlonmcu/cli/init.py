"""Command line subcommand for initializing a mlonmcu environment."""

import sys
import os
import pkg_resources
import pkgutil
import jinja2

from mlonmcu.environment.templates import all_templates
from mlonmcu.environment.config import get_environments_dir, DEFAULTS, env_subdirs
from mlonmcu.environment.util import in_virtualenv
from mlonmcu.environment.init import initialize_environment


def get_parser(subparsers):
    """ "Define and return a subparser for the init subcommand."""
    parser = subparsers.add_parser(
        "init", description="Initialize ML on MCU environment."
    )
    parser.set_defaults(func=handle)
    parser.add_argument(
        "-n",
        "--name",
        metavar="NAME",
        nargs=1,
        type=str,
        default="",
        help="Environment name (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--template",
        metavar="TEMPLATE",
        nargs=1,
        choices=all_templates.keys(),
        default=DEFAULTS["template"],
        help="Environment template (default: %(default)s, allowed: %(choices)s)",
    )
    parser.add_argument(
        "DIR",
        nargs="?",
        type=str,
        default=get_environments_dir(),
        help="Environment directory (default: "
        + os.path.join(get_environments_dir(), "{NAME}")
        + ")",
    )
    parser.add_argument(
        "--non-interactive",
        dest="non_interactive",
        action="store_true",
        help="Do not ask questions interactively",
    )
    parser.add_argument(
        "--venv",
        default=None,
        action="store_true",
        help="Create virtual python environment",
    )
    parser.add_argument(
        "--clone-models",
        default=None,
        dest="clone_models",
        action="store_true",
        help="Clone models directory into environment",
    )
    parser.add_argument(
        "--register",
        default=None,
        action="store_true",
        help="Add environment to the list of environment for the current user",
    )
    return parser


def handle(args):
    """Callback function which will be called to process the init subcommand"""
    name = args.name[0] if isinstance(args.name, list) else args.name
    initialize_environment(
        args.DIR,
        name,
        create_venv=args.venv,
        interactive=not args.non_interactive,
        clone_models=args.clone_models,
        register=args.register,
        template=args.template,
    )
