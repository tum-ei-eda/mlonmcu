# ML on MCU


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


Make sure to use a fresh virtual Python environment in the following steps.

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
### Usage

Is is recommended to checkout the provided [Demo Jupyter Notebook](https://github.com/tum-ei-eda/mlonmcu/blob/main/ipynb/Demo.ipynb) as it contains a end-to-end example which should help to understand the main concepts and methodology of the tool. The following paragraphs can be seen as a TL;DL version of the information in that Demo notebook.

While some tools and features of this project work out of the box, some of them require setting up an environment where additional dependencies are installed. This can be achived by creating a MLonMCU environment as follows:

```bash
mlonmcu init --name <environment_name> --template tgc # this creates a mlonmcu environment
```

Now you have to point the environment variable to your environment
```bash
export MLONMCU_HOME=<path_to_your_environment_variable>
mlonmcu env # this command lists all your available environments
```

The next step can take some time
```bash
mlonmcu setup
```
Now you are ready to go. For Usage it is best to checkout the Demo Jupyter notebook. The general flow will be described here.

An mlonmcu environment comes with a model zoo. If you want to implement your own model you can add it to the model directory of your environment. 
Be careful for each model that you add you need a single repository that is called the same as your .tflite model or the mlonmcu framework won't see your model
```bash
cd $MLONMCU_HOME/models # navigate to your environment model directory
mkdir <your_model_name>
cp <path_to_your_model>/<your_model_name>.tflite $MLONMCU_HOME/model/<your_model_name>/<your_model_name>.tflite #
mlonmcu models # to list all available models 
```

The general flow of a mlonmcu run looks like this. This is just a simple run that takes runs with self generated data
```bash
mlonmcu flow run model -b tvmaot -t tgc # model for your model name, -t for your target, -b for the backen you want its between tvm/tvmaot and tflmi
```
For further use and more usecases check the notebooks.
