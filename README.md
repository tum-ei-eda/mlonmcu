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

* TODO

## Getting started

While some tools and features of this project work out of the box, some of them require setting up an environment where additional dependencies are installed.

By default, this environment would be created intactively using the `mlonmcu init` command.

To use the created environment in a python program, a `MlonMcuContext` needs to be created as follows:

```
import mlonmcu.context

with mlonmcu.context.MlonMcuContext() as context:
    pass
```

The following table shows which parts of `mlonmcu` require such an context as a prerequisite:

TODO

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
