# Docker Images for MLonMCU

## Prerequisites

In addition to a regular Docker installation, `buildx` is required for building the MLonMCU images.

```sh
sudo apt install docker-buildx-plugin
```

## Container/Image Types

### CMake
TODO

### Minimal (CI)
TODO

### Default (User)
TODO

### Large (Benchmarking)
TODO

## Usage

### Command line

You can build the images as follows:

```sh
# Execute from top level of repository (here: main branch)

# Build all layers (takes a long time)
docker buildx build . -f docker/Dockerfile --tag tumeda/mlonmcu-bench:default-main-custom --build-arg ENABLE_CMAKE=true --build-arg MLONMCU_TEMPLATE=default

# Build default layers (recommended)
TODO
```

To use the previously built images, run the following commands

```sh
TODO
```

### Docker Compose

TODO
