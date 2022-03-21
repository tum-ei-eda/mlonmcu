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
import itertools

from mlonmcu.platform import get_platforms
from mlonmcu.config import resolve_required_config


def get_platform_names(context):
    return context.environment.lookup_platform_configs(names_only=True)


def get_platforms_targets(context):
    platform_names = get_platform_names(context)
    platform_classes = get_platforms()
    # To initialize the platform we need to provide a config with required paths.
    platforms = []
    for platform_name in platform_names:
        platform_cls = platform_classes[platform_name]
        required_keys = platform_cls.REQUIRED
        config = resolve_required_config(
            required_keys,
            cache=context.cache,
        )
        platform = platform_cls(config=config)
        platforms.append(platform)
    return {platform.name: platform.get_supported_targets() for platform in platforms}


def print_platforms(platform_names):
    print("Platforms:")
    if len(platform_names) == 0:
        print("No platforms found")
        return
    for name in platform_names:
        print("  - " + name)


def print_targets(platform_targets):
    print("Targets:")
    target_names = set(itertools.chain(*platform_targets.values()))
    if len(target_names) == 0:
        print("No targets found")
        return
    for target_name in target_names:
        target_platforms = [
            platform_name
            for platform_name, platform_targets in platform_targets.items()
            if target_name in platform_targets
        ]
        print("  - " + target_name + " [" + ", ".join(target_platforms) + "]")


def print_summary(context):
    print("Platform Targets Summary\n")
    platform_targets = get_platforms_targets(context)
    print_platforms(platform_targets.keys())
    print()
    print_targets(platform_targets)
