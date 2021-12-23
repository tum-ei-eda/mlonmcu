"""Command line subcommand for initializing a mlonmcu environment."""

import sys
import os
import xdg
import git
import pkg_resources
import pkgutil
import jinja2

from mlonmcu.config.templates import all_templates

config_dir = os.path.join(xdg.XDG_CONFIG_HOME, "mlonmcu")
environments_dir = os.path.join(config_dir, "environments")
environments_file = os.path.join(config_dir, "environments.txt")


#print("name", __name__)
#print("template_files", template_files)
##print("templates_dir", pkg_resources.resource_filename("mlonmcu", "../templates/" + template_files[0]))
#print("templates_dir", data.decode("utf-8"))

def get_template_names():
    template_files = pkg_resources.resource_listdir("mlonmcu", "../templates")
    names = [name.split(".yml.j2")[0] for name in template_files]
    return names


def fill_template(name, data={}):
    data = pkgutil.get_data("mlonmcu", os.path.join("..", "templates", name + ".yml.j2"))
    if data:
        text = data.decode("utf-8")
        tmpl = jinja2.Template(text)
        rendered = tmpl.render(home_dir=data["home_dir"], config_dir=data["config_dir"])
        #return None # rendered
    return None

#print("def", fill_template("default"))

def get_environments_map():
    result = {}
    if os.path.isfile(environments_file):
        with open(environments_file) as envs_file:
            if len(line) > 0:
                for line in envs_file:
                    temp = line.split("=")
                    assert len(lemp) == 2, "Invalid syntax found in environments file"
                    name, path = temp[:2]
                    if name in result:
                        raise RuntimeError("Found a duplicate environment name in environments file")
                    result[name] = path
    return result


def validate_name(name):
    # TODO: regex for valid names without spaces etc
    return True

def get_environment_names():
    envs_dict = get_environments_map()
    return envs_dict.keys()


def get_alternative_name(name, names):
    current = name
    i = -1
    while current in names:
        i = i + 1
        if i == 0:
            current = current + "_" + str(i)
        else:
            temp = current.split("_")
            current = "".join(temp[:-1]) + "_" + str(i)
    return current


def register_environment(name, path):
    validate_name(name)
    if not os.path.isabs(path):
        raise RuntimeError("Not an absolute path!")

    if not os.path.isfile(environments_file):
        raise RuntimeError("Environments file does not yet exist")

    with open(environments_file, "a") as envs_file:
        envs_file.write(name + "=" + path)


def write_environment_from_template():
    pass

def create_environment_directories(path, directories):
    if not os.path.isdir(path):
        raise RuntimeError(f"Not a diretory: {path}")
    for directory in directories:
        os.mkdirs(os.path.join(path, directories))


def clone_models_repo(dest):
    git.Git(dest).clone("https://github.com/PhilippvK/mlonmcu-models.git")  # FIXME: update url


DEFAULTS = {
    "environment": "default",
    "template": "default",
}

# TODO: move to different file
def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix

def in_virtualenv():
    """Detects if the current python interpreter is from a virtual environment."""
    return get_base_prefix_compat() != sys.prefix

def get_parser(subparsers):
    """"Define and return a subparser for the init subcommand."""
    parser = subparsers.add_parser('init', description='Initialize ML on MCU environment.')
    parser.set_defaults(func=handle)
    parser.add_argument('-n', '--name', metavar="NAME", nargs=1, type=str, default="",
                        help="Environment name (default: %(default)s)")
    parser.add_argument('-t', '--template', metavar="TEMPLATE", nargs=1,
                        choices=all_templates.keys(), default=DEFAULTS["template"],
                        help="Environment template (default: %(default)s, allowed: %(choices)s)")
    parser.add_argument('DIR', nargs='?', type=str, default=environments_dir,
                        help="Environment directory (default: " + os.path.join(environments_dir, "{NAME}") + ")")
    return parser

def handle(args):
    """Callback function which will be called to process the init subcommand"""
    print("Initializing ML on MCU environment")
    use_default_dir = (args.DIR == environments_dir)
    name = args.name[0] if isinstance(args.name, list) else args.name
    has_name = len(name.strip()) > 0
    if has_name:
        final_name = name.strip()
    else:
        if use_default_dir:
            final_name = DEFAULTS["environment"]
        else:
            final_name = "unamed"

    if use_default_dir:
        target_dir = os.path.join(args.DIR, final_name)
    else:
        target_dir = args.DIR
    target_dir = os.path.abspath(target_dir)
    print("Selected target directory:", target_dir)
    if os.path.exists(target_dir):
        print("The directory already exists!")
        if len(os.listdir(target_dir) ) > 0:
            print("The directory is not empty!", end=" - ")
            # TODO: check for mlonmcu project files, if yes ask for overwrite instead
            answer = input("Use anyway? [y/N]")
            if answer not in ["y", "Y"]:
                print("Aborting...")
                sys.exit(1)

    else:
        print("The directory does not exist!", end=" - ")
        answer = input("Create directory? [Y/n]")
        if answer not in ["n", "N"]:
            print("Aborting...")
            sys.exit(1)
    print(f"Creating environment.yml based on template '{args.template}'.")

    # TODO: create and maintain environments.yml in user directory?
    write_environment_from_template(args.template)

    # FIXME: controversial?
    if not in_virtualenv():
        print("It is strongly recommended to use mlonmcu inside a virtual Python environment.")
        input("Create one automatically? [Y/n]")
        # TODO: create venv
        venv_dir = os.path.join(target_dir, "venv")
        print(f"Virual environment was created in {venv_dir}. Make sure to activate it before using mlonmcu.")
    else:
        print("Skipping creation of virtual environment.")


    env_subdirs = ["deps"]
    answer = input("Clone mlonmcu-models repository into environment? [Y/n]")
    if answer not in ["n", "Y"]:
        clone_models_repo(os.path.join(target_dir, "models"))
    else:
        env_subdirs.append("models")
    print("Initializing directories in environment:", " ".join(env_subdirs))
    create_environment_directories(target_dir, env_subdirs)

    answer = input("Should the new environment be added to your list of environments? [Y/n]")
    if answer not in ["n", "N"]:
        if not os.path.isdir(config_dir):
            print(f"Config directory ({config_dir}) does not exist!", end=" - ")
            answer = input("Create it? [Y/n]")
            if answer not in ["n", "N"]:
                os.makedirs(config_dir)
        if not os.path.isfile(environments_file):
            print(f"Environments file ({environments_file}) does not exist!", end=" - ")
            answer = input("Create empty one? [Y/n]")
            if answer not in ["n", "N"]:
                open(environments_file, 'a').close()

        env_names = get_environment_names()
        if final_name in env_names:
            alternative_name = get_alternative_name(final_name, env_names)
            print(f"An environment with the name '{full_name}' already exists. Using '{alternative_name}' instead")
            final_name = alternative_name
        register_environment(final_name, target_dir)

    # TODO:
    print(f"Finished. Please add `export MLONMCU_HOME={target_dir}` to your shell configuration to use it anywhere")
