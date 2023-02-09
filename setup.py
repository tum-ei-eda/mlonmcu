#!/usr/bin/env python
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

"""The setup script."""

import os
from setuptools import setup, find_packages
import mlonmcu.setup.gen_requirements as gen_requirements
from mlonmcu import __version__


def resource_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = gen_requirements.join_requirements()
extra_require = {piece: deps for piece, (_, deps) in requirements.items() if piece not in ("all", "core")}

test_requirements = []

setup(
    author="TUM Department of Electrical and Computer Engineering - Chair of Electronic Design Automation",
    author_email="philipp.van-kempen@tum.de",
    python_requires=">=3.7",
    classifiers=[  # TODO
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="This project contains research code related to the deployment of inference"
    "or learning applications on tiny micro-controllers.",
    entry_points={
        "console_scripts": [
            "mlonmcu=mlonmcu.cli.main:main",
        ],
    },
    install_requires=requirements["core"][1],
    extras_require=extra_require,
    license="Apache License 2.0",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="mlonmcu",
    name="mlonmcu",
    packages=find_packages(include=["mlonmcu", "mlonmcu.*"]),
    package_data={"mlonmcu": resource_files("resources")},
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/tum-ei-eda/mlonmcu",
    version=__version__,
    zip_safe=False,
)
