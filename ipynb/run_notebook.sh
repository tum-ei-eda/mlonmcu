#!/bin/bash

NOTEBOOK=$1
DIRECTORY=$(dirname $NOTEBOOK)
YAML=$DIRECTORY/environment.yml.j2
REQUIREMTS=

mlonmcu init -t



python -m jupyter nbconvert --to notebook --execute AnalyseInstructionsFeature.ipynb
