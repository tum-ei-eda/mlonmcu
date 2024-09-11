# ML on MCU

[![pypi package](https://badge.fury.io/py/mlonmcu.svg)](https://pypi.org/project/mlonmcu)
[![readthedocs](https://readthedocs.org/projects/mlonmcu/badge/?version=latest)](https://mlonmcu.readthedocs.io/en/latest/?version=latest)
![coverage](https://byob.yarr.is/tum-ei-eda/mlonmcu/coverage)
[![GitHub license](https://img.shields.io/github/license/tum-ei-eda/mlonmcu.svg)](https://github.com/tum-ei-eda/mlonmcu/blob/main/LICENSE)

[![cicd workflow](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/cicd.yml/badge.svg)](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/cicd.yml)
[![lint workflow](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/style.yml/badge.svg)](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/style.yml)
[![demo workflow](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/demo.yml/badge.svg)](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/demo.yml)
[![bench workflow](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/bench.yml/badge.svg)](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/bench.yml)


This project contains research code related to the deployment of inference or learning applications on tiny micro-controllers.


* Free software: Apache License, Version 2.0
* Python Package: [https://pypi.org/project/mlonmcu/](https://pypi.org/project/mlonmcu/)
* Documentation: [https://mlonmcu.readthedocs.io](https://mlonmcu.readthedocs.io) or [https://tum-ei-eda.github.io/mlonmcu/](https://tum-ei-eda.github.io/mlonmcu/)


## Features

- Highly configurable python package
- Automatic resolution and installation of dependencies
- Supporting a large combination of frameworks/backends/targets/features
- Build-in parallel processing of large number of benchmarks
- Isolated enironments (not interfering with other installations)
- Command Line and Python Development Interfaces
- Docker images to get started quickly
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
sudo apt install libboost-all-dev graphviz doxygen libtinfo-dev zlib1g-dev texinfo unzip device-tree-compiler tree g++

# Optional (depending on configuration)
sudo apt install ninja-build flex lsb-release libelf-dev
```

Also make sure that your default Python is at least v3.7. If the `python` command is not available in your shell or points Python v2.7 check out `python-is-python3`.

**Warning:** It seems like the ETISS tool fails to compile if if find a version of LLVM 11 on your system which does not include Clang 11. The best workaround for now is to (if possible) remove those tools from your system: `sudo apt remove llvm-11* clang-11*` (See issue #1)

Make sure to use a fresh virtual Python environment in the following steps.

##### Install Release from PyPI

**Warning:** As the PyPI package is not always up to date, it is currently recommented to use a self-build version of the package (as explained in the next section)

To use the PIP package, run the following: `pip install mlonmcu` (Add `--user` if you are not using a virtual environment)


#### Build Package manually

First, install all relevant dependencies:

```
python -m venv .venv  # Feel free to choose a different directory or use a conda environment

# Run this whenever your have updated the repository
source .venv/bin/activate

# Environment-specific dependencies are installed later

**Warning:** It is recommended to have at least version 3.20 of CMake installed for full compatibility!

# Install ptional dependecies (only for development)
pip install -r requirements_dev.txt
pip install -r docs/requirements.txt

# Only if you want to use the provided python notebooks, as explained in  ./ipynb/README.md
pip install -r ipynb/requirements.txt
```

Then you should be able to install the `mlonmcu` python package like this

```
# Optionally remove an older version first: pip uninstall mlonmcu

make install  # Alternative: python setup.py install
```

#### Docker (Any other OS)

See [`./docker/README.md`](https://github.com/tum-ei-eda/mlonmcu/blob/main/docker/README.md) for more details.

This repository ships three different types of docker images based on Debian:

- A minimal one with preinstalled software dependencies and python packages

  Feel free to use this one if you do not want to install anything (except Docker) on your main sytem to work with mlonmcu
- A medium one which already has the `mlonmcu` python package installed

  Recommended and the easiest to use. (Especially when using `docker-compose` to mount shared directories etc.)

- A very large one with an already initialized and installed

  Mainly used for triggering automated benchmarks without spending too much time on downloading/compiling heavy dependencies over and over again.

### Usage

Is is recommended to checkout the provided [Demo Jupyter Notebook](https://github.com/tum-ei-eda/mlonmcu/blob/main/ipynb/Demo.ipynb) as it contains a end-to-end example which should help to understand the main concepts and methodology of the tool. The following paragraphs can be seen as a TL;DL version of the information in that Demo notebook.

While some tools and features of this project work out of the box, some of them require setting up an environment where additional dependencies are installed. This can be achived by creating a MLonMCU environment as follows:

```bash
mlonmcu init
```

Make sure to point the `MLONMCU_HOME` environment variable to the location of the previously initialied environment. (Alternative: use the `default` environment or `--home` argument on the command line)

Next, generate a `requirements_addition.txt` file inside the environment directory using `mlonmcu setup -g` which now be installed by running `pip install -r $MLONMCU_HOME/requirements_addition.txt` inside the virtual Python environment.


To use the created environment in a python program, a `MlonMcuContext` needs to be created as follows:

```
import mlonmcu.context

with mlonmcu.context.MlonMcuContext() as context:
    pass
```

## List of interesting MLonMCU forks

- MINRES TGC support: https://github.com/Minres/mlonmcu/tree/develop

## List of existing MLonMCU extensions/plugins

- ABC Example Plugin: coming soon!
- MINRES TGC Support: coming soon!

## Development

Make sure to first install the additonal set of development Python packages into your virtual environment:

```
pip install -r requirements_all.txt  # Install packages for every component (instead of using mlonmcu setup -g)
pip install -r requirements_dev.txt  # Building distributions and running tests
pip install -r docs/requirements.txt  # For working with the documentation
```

Unit test and integration test are defined in the `tests/` directory and can be triggered using `make test` or `pytest tests/`

Coverage can be determined by running `make coverage`. The latest coverage report (HTML) for the default branch can also be found as an artifact of the CI/CD workflow.

Documentation is mainly generated automatically from doctrings (triggered via `make html`). It is also possible to include markdown files from the repo into the `.rst` files found in the [`docs/`](./docs/) directory. There is a GitHub workflow which publishes the documentation for the default branch to our [GitHub Pages](https://tum-ei-eda.github.io/mlonmcu).

Regarding coding style, it is recommended to run `black` before every commit. The default line length should be given in the `setup.cfg` file.

### Developers

- Rafael Stahl (TUM) [@rafzi]

  - Wrote initial version of the MLonMCU project

- Philipp van Kempen (TUM) [@PhilippvK]

  - Came up with MLonMCU Python package

## Publications

- **MLonMCU: TinyML Benchmarking with Fast Retargeting** ([https://dl.acm.org/doi/10.1145/3637543.3652878](https://dl.acm.org/doi/abs/10.1145/3615338.3618128))

  *CODAI '23: Proceedings of the 2023 Workshop on Compilers, Deployment, and Tooling for Edge AI*

  BibTeX

  ```bibtex
  @inproceedings{10.1145/3615338.3618128,
    author = {van Kempen, Philipp and Stahl, Rafael and Mueller-Gritschneder, Daniel and Schlichtmann, Ulf},
    title = {MLonMCU: TinyML Benchmarking with Fast Retargeting},
    year = {2024},
    isbn = {9798400703379},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3615338.3618128},
    doi = {10.1145/3615338.3618128},
    abstract = {While there exist many ways to deploy machine learning models on microcontrollers, it is non-trivial to choose the optimal combination of frameworks and targets for a given application. Thus, automating the end-to-end benchmarking flow is of high relevance nowadays. A tool called MLonMCU is proposed in this paper and demonstrated by benchmarking the state-of-the-art TinyML frameworks TFLite for Microcontrollers and TVM effortlessly with a large number of configurations in a low amount of time.},
    booktitle = {Proceedings of the 2023 Workshop on Compilers, Deployment, and Tooling for Edge AI},
    pages = {32–36},
    numpages = {5},
    keywords = {TinyML, neural networks, microcontrollers},
    location = {Hamburg, Germany},
    series = {CODAI '23}
  }
  ```

### Other
This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template. However most of the templates was manually changed to be in Markdown instead of reStructuredText.

- **Cookiecutter:** https://github.com/audreyr/cookiecutter
- **`audreyr/cookiecutter-pypackage`:** https://github.com/audreyr/cookiecutter-pypackage


## Acknowledgment

<img src="./BMBF_gefoerdert_2017_en.jpg" alt="drawing" height="75" align="left" >

This research is partially funded by the German Federal Ministry of Education and Research (BMBF) within
the projects [Scale4Edge](https://www.edacentrum.de/scale4edge/) (grant number 16ME0127) and [MANNHEIM-FlexKI](https://www.edacentrum.de/projekte/MANNHEIM-FlexKI) (grant number 01IS22086L).
