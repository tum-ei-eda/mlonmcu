# ML on MCU

[![](https://img.shields.io/pypi/v/mlonmcu.svg)](https://pypi.python.org/pypi/mlonmcu)
[![](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/cicd.yml/badge.svg)](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/cicd.yml)
[![](https://readthedocs.org/projects/mlonmcu/badge/?version=latest)](https://mlonmcu.readthedocs.io/en/latest/?version=latest)
![](https://byob.yarr.is/tum-ei-eda/mlonmcu/coverage)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/)
[![GitHub license](https://img.shields.io/github/license/tum-ei-eda/mlonmcu.svg)](https://github.com/tum-ei-eda/mlonmcu/blob/main/LICENSE)

This project contains research code related to the deployment of inference or learning applications on tiny micro-controllers.


* Free software: MIT license
* Documentation: https://mlonmcu.readthedocs.io.


## Features

- Highly configurable python package
- Automatic resolution and installation of dependencies
- Supporting a large combination of frameworks/backends/targets/features:
  - Frameworks (Backenss): TFLite Micro, MircoTVM
  - Targets: Host (x86), ETISS (Pulpino)
  - Features: Autotuning (WIP), Debugging
- Build-in parallel processing of large number of benchmarks
- Isolated enironments (not interfering with other installations)
- Command Line and Python Development Interfaces
- WIP: Docker images to get started quickly
- Extensive documentation on usage and code details
- CI/CD integration and high PyTest coverage

## Getting started

### Prerequisites

#### Ubuntu/Debian

First, a set of APT packages needs to be installed:

```
# Python related
sudo apt install python3-pip python3-venv

# MLonMCU related
sudo apt install libboost-system-dev libboost-filesystem-dev libboost-program-options-dev graphviz doxygen libtinfo-dev zlib1g-dev texinfo unzip device-tree-compiler tree g++

# Optional (depending on configuration)
sudo apt install ninja-build
```

Also make sure that your default Python is at least v3.7. If the `python` command is not available in your shell or points Python v2.7 check out `python-is-python3`.

**Warning:** It seems like the ETISS tool fails to compile if if find a version of LLVM 11 on your system which does not include Clang 11. The best workaround for now is to (if possible) remove those tools from your system: `sudo apt remove llvm-11* clang-11*` (See issue #1)

Next, make sure to install all package dependencies into a fresh virtual Python environment

```
python -m venv .venv  # Feel free to choose a different directory or use a conda environment

# Run this whenever your have updated the repository
source .venv/bin/activate
pip install -r requirements.txt

**Warning:** It is recommended to have at least version 3.20 of CMake installed for full compatibility!

# Only if you want to use the provided python notebooks, as explained in  ./ipynb/README.md
pip install -r ipynb/requirements.txt

# Optional (only for development)
pip install -r requirements_dev.txt
pip install -r docs/requirements.txt
```

Then you should be able to install the `mlonmcu` python package like this

```
# Optionally remove an older version first: pip uninstall mlonmcu

make install  # Alternative: python setup.py install
```

#### Docker (Any other OS)

See [`./docker/README.md`](htps://github.com/tum-ei-eda/mlonmcu/blob/main/docker/README.md) for more details.

This repository ships three different types of docker images based on Debian:

- A minimal one with preinstalled software dependencies and python packages

  Feel free to use this one if you do not want to install anything (except Docker) on your main sytem to work with mlonmcu
- A medium one which already has the `mlonmcu` python package installed

  Recommended and the easiest to use. (Especially when using `docker-compose` to mount shared directories etc.)

- A very large one with an already initialized and installed

  Mainly used for triggering automated benchmarks without spending too much time on downloading/compiling heavy dependencies over and over again.

### Usage

Is is recommended to checkout the provided [Demo Jupyter Notebook](https://github.com/tum-ei-eda/mlonmcu/blob/main/ipynb/Demo.ipynb) as it contains a end-to-end example which should help to understand the main concepts and methodology of the tool. The following paragraphs can be seen as a TL;DL version of the information in that Demo notebook.

#### Using the command line

TODO

#### Using the Python API

TODO

While some tools and features of this project work out of the box, some of them require setting up an environment where additional dependencies are installed.

By default, this environment would be created intactively using the `mlonmcu init` command.

To use the created environment in a python program, a `MlonMcuContext` needs to be created as follows:

```
import mlonmcu.context

with mlonmcu.context.MlonMcuContext() as context:
    pass
```

The following table shows which parts of `mlonmcu` require such an context as a prerequisite:

### Important terms

- Context
- Environment
- Run
- Backend
- Feature
- Target
- (Flow)
- Framework
- Frontend
- Task

TODO
## Development

Make sure to first install the additonal set of development Python packages into your virtual environment:

```
pip install -r requirements_dev.txt  # Building distributions and running tests
pip install -r docs/requirements.txt  # For working with the documentation
```

Unit test and integration test are defined in the `tests/` directory and can be triggered using `make test` or `pytest tests/`

Coverage can be determined by running `make coverage`. The latest coverage report (HTML) for the default branch can also be found as an artifact of the CI/CD workflow.

Documentation is mainly generated automatically from doctrings (triggered via `make html`). It is also possible to include markdown files from the repo into the `.rst` files found in the [`docs/`](./docs/) directory. There is a GitHub workflow which publishes the documentation for the default branch to our [GitHub Pages](https://tum-ei-eda.github.io/mlonmcu).

Regarding coding style, it is recommended to run `black` before every commit. The default line length should be given in the `setup.cfg` file.

### Future Work

- [ ] Finish beta version `v0.1.0`
- [ ] Open Source Release
  - [ ] Make repository public
  - [ ] Release python package
  - [ ] Publish docs automatically to https://readthedocs.org

## Credits

This is a research project proposed by the Chair of Design Automation of the Technical University of Munich.

### Developers

- Rafael Stahl (TUM) [@rafzi]

  - Wrote initial version of the MLonMCU project

- Philipp van Kempen (TUM) [@PhilippvK]

  - Came up with MLonMCU Python package


### Other
This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template. However most of the templates was manually changed to be in Markdown instead of reStructuredText.

- **Cookiecutter:** https://github.com/audreyr/cookiecutter
- **`audreyr/cookiecutter-pypackage`:** https://github.com/audreyr/cookiecutter-pypackage
