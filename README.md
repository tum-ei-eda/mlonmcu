# ML on MCU

.. image:: https://img.shields.io/pypi/v/mlonmcu.svg
        :target: https://pypi.python.org/pypi/mlonmcu

.. image:: https://img.shields.io/travis/tum-ei-eda/mlonmcu.svg
        :target: https://travis-ci.com/tum-ei-eda/mlonmcu

.. image:: https://readthedocs.org/projects/mlonmcu/badge/?version=latest
        :target: https://mlonmcu.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


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
# mlonmcu
