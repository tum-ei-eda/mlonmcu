# Docker Images for MLonMCU

## Prerequisites

In addition to a regular Docker installation, `buildx` is required for building the MLonMCU images.

```sh
sudo apt install docker-buildx-plugin
```

## Container/Image Types

### CMake
Minimal image to add a recent version of CMake on old ubuntu versions. Not required anymore for Ubuntu 22.04+

### Minimal (CI)

The `mlonmcu-ci` image is an Ubuntu images with all the operating system packages required by MLonMCU and its dependencies.

### Default (User)

The `mlonmcu` image contains an installation of the MLonMCU Python package with all dependencies. An environment has to be created as a next step. This allows users to run MLonMCU commands inside a docker container while operating with environments stored on the host filesystem (using volumes/mounts).

### Large (Benchmarking)

The image `mlonmcu-bench` builds up on the user `mlonmcu` image by initializing and installing a specific environment on it. This allows to use the image for benchmarks and integration tests.

### Large (Benchmarking) - Updated

Incremental update of `mlonmcu-bench` image for daily rebuilds without setting up an environment from scratch. Effectively running git pull to refresh mlonmcu sources and `mlonmcu setup --rebuild` to refresh the dependencies.

## Usage

### Pulling Prebuilt Images

CMake/Minimal images are hosted in the GitHub container registry.

Due to their large size, all other images are hosted on Docker Hub.

All images are versioned by date. The `latest` tag is added to the most recent images.

#### [Cmake Image](https://github.com/tum-ei-eda/mlonmcu/pkgs/container/mlonmcu-cmake)
  
  Command: `docker pull ghcr.io/tum-ei-eda/mlonmcu-cmake:latest`

#### [Minimal/CI Image](https://github.com/tum-ei-eda/mlonmcu/pkgs/container/mlonmcu-ci)
  
  Command: `docker pull ghcr.io/tum-ei-eda/mlonmcu-ci:develop-20240420`

#### [Default/User Image](https://hub.docker.com/r/tumeda/mlonmcu)

  Note: none-default branches are added as a suffix to the tag.

  - Branch: `main` ![Docker Image Version (tag)](https://img.shields.io/docker/v/tumeda/mlonmcu/latest)
 ![Docker Image Size (tag)](https://img.shields.io/docker/image-size/tumeda/mlonmcu/latest)


    Command: `docker pull tumeda/mlonmcu:latest` 

  - Branch: `develop`  ![Docker Image Version (tag)](https://img.shields.io/docker/v/tumeda/mlonmcu/develop-latest)
 ![Docker Image Size (tag)](https://img.shields.io/docker/image-size/tumeda/mlonmcu/develop-latest)

    Command: `docker pull tumeda/mlonmcu:develop-latest`

#### [Large/Benchmarking Image](https://hub.docker.com/r/tumeda/mlonmcu-bench)

  Notes: These images are refreshed **weekly**. There also exist **daily** incremental builds with the sufffix `-updated` (i.e. `tumeda/mlonmcu-bench:dev-develop-latest-updated`). Non-default envionments are added as a prefix to the tag.

  - Branch: `main` , Environment: `dev` ![Docker Image Version (tag)](https://img.shields.io/docker/v/tumeda/mlonmcu-bench/latest)
 ![Docker Image Size (tag)](https://img.shields.io/docker/image-size/tumeda/mlonmcu-bench/latest)
  
    Command: `docker pull tumeda/mlonmcu-bench:latest`

  - Branch: `default` , Environment: `dev` ![Docker Image Version (tag)](https://img.shields.io/docker/v/tumeda/mlonmcu-bench/dev-develop-latest)
 ![Docker Image Size (tag)](https://img.shields.io/docker/image-size/tumeda/mlonmcu-bench/dev-develop-latest)
  
    Command: `docker pull tumeda/mlonmcu:develop-latest`

  - Branch: `develop`, Environment: `vicuna` ![Docker Image Version (tag)](https://img.shields.io/docker/v/tumeda/mlonmcu-bench/vicuna-develop-latest)
 ![Docker Image Size (tag)](https://img.shields.io/docker/image-size/tumeda/mlonmcu-bench/vicuna-develop-latest)
  
    Command: `docker pull tumeda/mlonmcu-bench:vicuna-develop-latest`

  - Branch: `develop`, Environment: `ara` ![Docker Image Version (tag)](https://img.shields.io/docker/v/tumeda/mlonmcu-bench/ara-develop-latest)
 ![Docker Image Size (tag)](https://img.shields.io/docker/image-size/tumeda/mlonmcu-bench/ara-develop-latest)
  
    Command: `docker pull tumeda/mlonmcu-bench:ara-develop-latest`

  - Branch: `develop`, Environment: `corev` ![Docker Image Version (tag)](https://img.shields.io/docker/v/tumeda/mlonmcu-bench/corev-develop-latest)
 ![Docker Image Size (tag)](https://img.shields.io/docker/image-size/tumeda/mlonmcu-bench/corev-develop-latest)
  
    Command: `docker pull tumeda/mlonmcu-bench:corev-develop-latest`



### Building Images

#### Available Dockerfiles

**[`Dockerfile`](https://github.com/tum-ei-eda/mlonmcu/blob/main/docker/Dockerfile)**

Used to build CMake/Minimal/Default/Large images. The images a split up to several stages to allow reuse.

**[`Dockerfile2`](https://github.com/tum-ei-eda/mlonmcu/blob/main/docker/Dockerfile2)**

Allows incremental builts on top of the images build by `Dockerfile`.

#### Command line (local)

You can build the images as follows:

```sh
# Execute from top level of repository (here: main branch)

# Build all layers (takes a long time)
docker buildx build . -f docker/Dockerfile --tag tumeda/mlonmcu-bench:default-main-custom --build-arg ENABLE_CMAKE=true --build-arg MLONMCU_TEMPLATE=default

# Build default layers (recommended)
TODO
```

#### Via GitHub Actions (online)

**Manual builds**

Use the `Run Workflow` button on https://github.com/tum-ei-eda/mlonmcu/actions/workflows/container.yml to build images for a specific `branch` & `environment`.

**Manual builds (incremental)**

Use the `Run Workflow` button on https://github.com/tum-ei-eda/mlonmcu/actions/workflows/refresh_container.yml. Specify the base image and output image as well as the environment template.

**Scheduled builds**

[Weekly jobs](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/container_weekly.yml)

```yaml
schedule:
  - cron: "0 0 * * 6"  # main, default
  - cron: "0 6 * * 6"  # develop, corev
  - cron: "0 12 * * 6"  # develop, dev
  - cron: "0 18 * * 6"  # develop, ara
  - cron: "0 20 * * 6"  # develop, vicuna
```

[Daily jobs](https://github.com/tum-ei-eda/mlonmcu/actions/workflows/refresh_container_daily.yml)

```yaml
schedule:
  - cron: "0 12 * * *"  # develop, dev
```

**CI Setup for DockerHub**

If you want to use the actions on a fork, they will fail due to missing docker credentials. Please export the followinf variables via `Settings -> Secrets -> Actions` to tell the jobs about your username and password:

- `DOCKER_PASSWORD`
- `DOCKER_USERNAME`

### Running the Images in Docker Containers

To use the previously built images, run the following commands

```sh
TODO
```

### Docker Compose

TODO
