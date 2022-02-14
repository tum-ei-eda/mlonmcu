ARG ENABLE_CMAKE="false"

FROM debian:bullseye AS cmake-install
ARG CMAKE_REPO="https://github.com/Kitware/CMake"
ARG CMAKE_VERSION="3.15.5"
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
    && apt-get -y install --no-install-recommends wget ca-certificates ; \
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
    g++ libboost-system-dev libboost-filesystem-dev libboost-program-options-dev graphviz doxygen libtinfo-dev zlib1g-dev verilator \
    autoconf automake autotools-dev curl python3 libmpc-dev libmpfr-dev libgmp-dev gawk build-essential \
    bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ca-certificates \
    virtualenv git python3-dev python3-pip python3-setuptools libtinfo5 \
    gcc libedit-dev libxml2-dev gcc-arm-none-eabi libopenblas-dev liblapack-dev gfortran cmake unzip ninja-build device-tree-compiler

CMD /bin/bash

FROM mlonmcu-ci AS mlonmcu

ADD . /mlonmcu
WORKDIR /mlonmcu
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install .
CMD mlonmcu

ARG MLONMCU_TEMPLATE=default
FROM mlonmcu AS mlonmcu-bench

WORKDIR /environment

RUN mlonmcu init --template ${MLONMCU_TEMPLATE} --non-interactive /environment --skip-sw
RUN cp -r /mlonmcu/models/* /environment/models && cp -r /mlonmcu/sw /environment/sw
ENV MLONMCU_HOME=/environment

RUN mlonmcu setup -v

CMD mlonmcu
VOLUME /environment
