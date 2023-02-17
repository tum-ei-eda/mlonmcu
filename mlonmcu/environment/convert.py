#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# import yaml
# import pathlib
# import logging
#
# from .config import PathConfig


import argparse
from mlonmcu.environment.environment import UserEnvironment
from mlonmcu.environment.environment2 import UserEnvironment2
from mlonmcu.environment.config import ComponentConfig, DefaultsConfig

# from mlonmcu.environment.loader import load_environment_from_file


def convert_v1_to_v2(env):
    frameworks = {f.name: ComponentConfig(f.enabled, f.name == env.defaults.default_framework) for f in env.frameworks}
    backends = {
        b.name: ComponentConfig(
            b.enabled, f.name == env.defaults.default_framework and b.name == env.defaults.default_backends[f.name]
        )
        for f in env.frameworks
        for b in f.backends
        if f.enabled
    }
    toolchains = {t: ComponentConfig(True, False) for t in env.toolchains}
    frontends = {f.name: ComponentConfig(f.enabled, f.enabled) for f in env.frontends}
    platforms = {p.name: ComponentConfig(p.enabled, p.enabled) for p in env.platforms}
    targets = {t.name: ComponentConfig(t.enabled, t.name == env.defaults.default_target) for t in env.targets}

    framework_features = {
        f_.name: ComponentConfig(f_.supported, False) for f in env.frameworks for f_ in f.features if f.enabled
    }
    backend_features = {
        f_.name: ComponentConfig(f_.supported, False)
        for f in env.frameworks
        for b in f.backends
        for f_ in b.features
        if b.enabled and f.enabled
    }
    target_features = {
        f_.name: ComponentConfig(f_.supported, False) for t in env.targets for f_ in t.features if t.enabled
    }
    platform_features = {
        f_.name: ComponentConfig(f_.supported, False) for p in env.platforms for f_ in p.features if p.enabled
    }
    frontend_features = {
        f_.name: ComponentConfig(f_.supported, False) for f in env.frontends for f_ in f.features if f.enabled
    }
    other_features = {}
    # print("framework_features", framework_features)
    # print("backend_features", backend_features)
    # print("target_features", target_features)
    # print("frontend_features", frontend_features)
    # print("other_features", other_features)
    # print("platform_features", platform_features)

    def helper(name):
        return any(
            [
                name in x and x[name].supported
                for x in [
                    framework_features,
                    backend_features,
                    target_features,
                    platform_features,
                    frontend_features,
                    other_features,
                ]
            ]
        )

    all_feature_names = (
        set(framework_features.keys())
        .union(set(backend_features.keys()))
        .union(set(target_features.keys()))
        .union(set(platform_features.keys()))
        .union(set(frontend_features.keys()))
        .union(set(other_features.keys()))
    )
    features = {name: ComponentConfig(helper(name), False) for name in all_feature_names}
    # print("features", features)
    # print("?")

    env2 = UserEnvironment2(
        env.home,
        alias=env.alias,
        defaults=DefaultsConfig(
            log_level=env.defaults.log_level,
            log_to_file=env.defaults.log_to_file,
            log_rotate=env.defaults.log_rotate,
            cleanup_auto=env.defaults.cleanup_auto,
            cleanup_keep=env.defaults.cleanup_keep,
        ),
        paths=env.paths,
        repos=env.repos,
        frameworks=frameworks,
        backends=backends,
        frontends=frontends,
        platforms=platforms,
        toolchains=toolchains,
        targets=targets,
        features=features,
        # postprocesses=None,
        variables=env.vars,
        default_flags=env.flags,
    )
    return env2


def main():
    parser = argparse.ArgumentParser(
        description=f"Converter for environment files/templates",
    )
    parser.add_argument("input", metavar="FILE", type=str, nargs=1, help="File to process")
    parser.add_argument(
        "--output",
        "-o",
        metavar="DEST",
        type=str,
        default=None,
        help="""Output directory/file (default: print to stdout instead)""",
    )
    parser.add_argument(
        "--in-version",
        metavar="VER",
        type=int,
        default=1,
        help="""Output directory/file (default: %(default)s)""",
    )
    parser.add_argument(
        "--out-version",
        metavar="VER",
        type=int,
        default=2,
        help="""Output directory/file (default: %(default)s)""",
    )
    args = parser.parse_args()

    assert args.in_version == 1 and args.out_version == 2, "Unsupported combination of versions"

    file = args.input[0]

    env = UserEnvironment.from_file(file)

    env2 = convert_v1_to_v2(env)

    if args.output:
        env2.to_file(args.output)
    else:
        print(env2.to_text())


if __name__ == "__main__":
    main()
