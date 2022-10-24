# Environments

## Motivation

While the base set of features in MLonMCU should work out of the box, there are some reasons for not sticking to predefined default values. Instead of hardcoding values such as repository urls or file paths inside the codebase, they can be completely configured by the user to allow setting up custom environments with very low efforts required. Having multiple MLonMCU environments installed in parallel has further advantages as they are completely isolated from each other an therefore allow using different versions of components and a different set of features. In addition there is a possibility to turn of certain components completely to reduce the installation time. User configuration for the MLonMCU flow which would typically need to be passed via the command line can be instead defined in the environment file which going to be explained next.

## `environment.yml` Explained

Each MLonMCU environment has a unique directory which can be chosen by the user where dependencies are installed and exports are written to. In this directory which can also be referred as MLonMCU-Home the environment configuration file `environment.yml` can be found. The basic structure of this YAML file can be summarized as follows:

```yaml
# The MLONMCU_HOME is filled in automatically when creating the environment
home: "{{ home_dir }}"
logging:
  enabled: false
  ...
# Default locations for certain directoriescan be changed here
# Non-absolute paths will always be threated relative to the MLONMCU_HOME
paths:
  deps: deps
  ...
# Here default clone_urls
repos:
  some_repo:
    url: "insert_repo_url_here"
    ref: optional_branch_tag_or_commit
  ...

# Here all supported frameworks with their specific features are defined
# Optionally disable unwanted or incomatible backends or features here
# The configured defaults are used if no backend was specified in the command line options
frameworks:
  default: some_framework
  some_framework:
    enabled: true
    backends:
      default: some_backend
      some_backend:
        enabled: true
        features:
          some_feature: true
          ...
      ...
    features:
      another_feature: true
      ...
  ...
# The enabled fronends are processed in the order defined here until a compatible one is found for a given model type
frontends:
  some_frontend:
    enabled: true
    features:
      some_feature: false
      ...
  another_frontend:
    enabled: false
  ...
# List of supported targets in the environment
targets:
  default: some_target
  some_target:
    enabled: true
    features:
      some_featuee: true
  ...
# This is where further options such as specific versions of dependencies can be set in the furture
vars:
  some_backend.some_var: 10
  foo: "bar"
```

While some parts of the file can theoretically be omitted, it is not recommended to do do. Also it has to be noted, that frameworks, backends, targets, frontends and features need to be explicitly enabled in the environment file to be available in the MLonMCU flow.

**Hint:** The `default` property which is available for some components supports wildcards, e.g. instead of providing a single backend name just put in "*" to use all enabled backends of the given framework by default.
## Environment Templates

There is a set of environment file templates provided with the MLonMCU package which can be chosen from by the user e.g.

- `default`: Should work out of the box for everyone
- `minimal`: Stripped down version of MLonMCU with only a small set of dependencies (just the essentials)
- `dev`: Development version which will is not guaranteed to work all the time.
- `tumeda`: Version on MLonMCU depending on tool which are (not yet) available publicly.

After a template was chosen, the initial environment file is being generated which can be freely modified by the user afterwards.

## Creating environments 

### Command line (recommended)

To get started with MLonMCU on the command line first an environment has to be created using the `mlonmcu init` command. As only some usage examples are shown in the following, make sure to check out `mlonmcu init --help` to learn more.

- Initialize a default environment at the default location (`~/.config/mlonmcu/environments/default` on most UNIX Systems): `mlonmcu init`
- Initialize an environment inside the current working directory: `mlonmcu init -H .`

The tool will ask some questions on the command line interactively.

### Python API

At the moment please stick to the CLI tool!

## Using environments

### Command line

Most of the `mlonmcu` subcommands need a MLonMCU environment to operate on. In some cases it can be resolved automatically however it is recommended to pass it explicitly by the user in either of the following ways:

- Point the `MLONMCU_HOME` environment variable to the environment directory which should be used.
- Use the command line flags `-H` (`--home` or `--hint`) to provide either the path or (if available) the registered name of the environment.

If none of this was specified, MLonMCU will first look for a valid environment in the current working directory and else fall back to the default environment of the current user (if configured).

Example usage:

```
export MLONMCU_HOME=/tmp/home
mlonmcu models
```

or

```
mlonmcu models -H myenv
```

or

```
mlonmcu models -H ./env/
```


### Python API

For the best experience a MLonMCU environment should always we wrapped with a `MlonMcuContext` as it provides useful utilities and a locking mechanism which can ensure that only one instance of the environment can be requested at a time.

The typical using using a Python `with` block looks as follows:

```python
from mlonmcu.context.context import MlonMcuContext

with MlonMcuContext() as ctx:
    pass
```

Analogous to the command line flags an environment path or name should be provided to use a non-default environment location.

## Environment registry

Optionally environment can be registered in the users home config directory with a given name which enabled referring to them without providing a file path. Use the `--register` and `--name` flags of the `mlonmcu init` command to do so. The command `mlonmcu env` provides useful utilities to list and modify existing entries in the registry file which is typically located at `~/.config/mlonmcu/environments.ini`.
