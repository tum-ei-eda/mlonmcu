##
## Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
##
## This file is part of MLonMCU.
## See https://github.com/tum-ei-eda/mlonmcu.git for further info.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
ARG ENABLE_CMAKE="false"

FROM debian:bullseye AS cmake-install
ARG CMAKE_REPO="https://github.com/Kitware/CMake"
ARG CMAKE_VERSION="3.22.2"
ARG CMAKE_TARGET="Linux-x86_64"
ARG ENABLE_CMAKE

RUN mkdir -p /opt/tools/cmake  \
    && if [ "${ENABLE_CMAKE}" = "true" ] ; \
    then echo "Installing CMake..." ; \
    else echo "Skipping cmake-install" ; \
    fi

RUN if [ "${ENABLE_CMAKE}" = "true" ] ; \
    then apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends wget ca-certificates \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/ \
    && rm -rf /var/cache/apt/lists; \
    fi

# Install modern CMake
RUN if [ "${ENABLE_CMAKE}" = "true" ] ; \
    then cd /tmp \
    && wget -O cmake.sh ${CMAKE_REPO}/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-${CMAKE_TARGET}.sh \
    && sh ./cmake.sh --prefix=/opt/tools/cmake/ --skip-license \
    && rm ./cmake.sh ; \
    fi

FROM debian:bullseye AS mlonmcu-ci

COPY --from=cmake-install /opt/tools/cmake/ /opt/tools/cmake/
ENV PATH="/opt/tools/cmake/bin:${PATH}"
ENV CMAKE_DIR="/opt/tools/cmake"

# Install APT dependencies  (TODO: update list)
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends wget unzip vim openssh-client \
    g++ graphviz doxygen libtinfo-dev zlib1g-dev \
    autoconf automake autotools-dev curl python3 libmpc-dev libmpfr-dev libgmp-dev gawk build-essential \
    bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ca-certificates \
    virtualenv git python3-dev python3-pip python3-setuptools python3-venv libtinfo5 libzstd-dev \
    gcc libedit-dev libxml2-dev libopenblas-dev liblapack-dev gfortran cmake unzip ninja-build python-is-python3 device-tree-compiler libboost-all-dev \
    libfl-dev lsb-release libelf-dev \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/ \
    && rm -rf /var/cache/apt/lists

RUN dpkg -l

CMD ["/bin/bash"]

FROM mlonmcu-ci AS mlonmcu

ADD . /mlonmcu
WORKDIR /mlonmcu
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN . /venv/bin/activate && pip install --upgrade pip --no-cache-dir
RUN . /venv/bin/activate && pip install -r requirements.txt --no-cache-dir
RUN . /venv/bin/activate && pip install . --no-cache-dir

CMD ["mlonmcu"]

ARG MLONMCU_TEMPLATE=default
FROM mlonmcu AS mlonmcu-bench

WORKDIR /environment

RUN mlonmcu init --template ${MLONMCU_TEMPLATE} --non-interactive /environment --clone-models
ENV MLONMCU_HOME=/environment

RUN mlonmcu setup -g

RUN . /venv/bin/activate && pip install -r /environment/requirements_addition.txt --no-cache-dir

RUN df -h

RUN df -h

RUN . /venv/bin/activate && mlonmcu setup -v

RUN df -h

CMD ["mlonmcu"]
VOLUME /environment
