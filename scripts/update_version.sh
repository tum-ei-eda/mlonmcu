#!/bin/bash

set -e

DEV_VERSION=$(git describe --tags --match "v[0-9]*.[0-9]*.[0-9]*" --match "v[0-9]*.[0-9]*.dev[0-9]*" | cut -d'-' -f1-2 | sed 's/0\-//')

head -n -1 mlonmcu/version.py > mlonmcu/version.py.new
echo "__version__ = \"${DEV_VERSION:1}\"" >> mlonmcu/version.py.new
mv mlonmcu/version.py.new mlonmcu/version.py
