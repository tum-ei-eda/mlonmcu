#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Definitions of mlonmcu config templates."""
import pkgutil
import os
import jinja2
from pathlib import Path
import pkg_resources

from .config import get_config_dir


def get_template_names():
    template_files = pkg_resources.resource_listdir("mlonmcu", os.path.join("..", "resources", "templates"))
    names = [name.split(".yml.j2")[0] for name in template_files]
    return names


def get_template_text(name):
    return pkgutil.get_data("mlonmcu", os.path.join("..", "resources", "templates", name + ".yml.j2"))


def fill_template(name, data={}):
    if name.endswith(".j2"):  # Template from file
        assert Path(name).is_file(), f"Template does not exits: {name}"
        with open(name, "r") as handle:
            template_text = handle.read()
    else:  # Template by name
        template_text = get_template_text(name)
    if template_text:
        try:
            template_text = template_text.decode("utf-8")
        except (UnicodeDecodeError):
            pass
        tmpl = jinja2.Template(template_text)
        rendered = tmpl.render(**data)
        return rendered
    return None


def fill_environment_yaml(template_name, home_dir):
    return fill_template(template_name, {"home_dir": str(home_dir), "config_dir": str(get_config_dir())})


def write_environment_yaml_from_template(path, template_name, home_dir):
    with open(path, "w") as yaml:
        text = fill_environment_yaml(template_name, home_dir)
        yaml.write(text)
