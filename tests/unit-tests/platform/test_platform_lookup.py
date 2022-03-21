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
import mock
import mlonmcu.platform.platform
from mlonmcu.platform.platform import TargetPlatform
import mlonmcu.platform.lookup


# TODO: also support non-target platforms and show their type in the list


class MyPlatformA(TargetPlatform):
    def __init__(self, features=None, config=None):
        super().__init__(
            "platform_a",
            features=features,
            config=config,
        )


class MyPlatformB(TargetPlatform):
    def __init__(self, features=None, config=None):
        super().__init__(
            "platform_b",
            features=features,
            config=config,
        )

    def get_supported_targets(self):
        return ["target_1"]


class MyPlatformC(TargetPlatform):
    def __init__(self, features=None, config=None):
        super().__init__(
            "platform_c",
            features=features,
            config=config,
        )

    def get_supported_targets(self):
        return ["target_1", "target_2"]


def test_platform_print_summary(capsys, fake_context):
    # empty
    with mock.patch("mlonmcu.platform.lookup.get_platform_names", return_value=[]):
        with mock.patch("mlonmcu.platform.platform.get_platforms", return_value={}):
            mlonmcu.platform.lookup.print_summary(fake_context)
            out, err = capsys.readouterr()
            split = out.split("\n")
            assert "No platforms found" in split
            assert "No targets found" in split

    # single platform, no targets
    with mock.patch("mlonmcu.platform.lookup.get_platform_names", return_value=["platform_a"]):
        with mock.patch("mlonmcu.platform.lookup.get_platforms", return_value={"platform_a": MyPlatformA}):
            mlonmcu.platform.lookup.print_summary(fake_context)
            out, err = capsys.readouterr()
            split = out.split("\n")
            assert "  - platform_a" in split
            assert "No targets found" in split

    # single platform, single target
    with mock.patch("mlonmcu.platform.lookup.get_platform_names", return_value=["platform_b"]):
        with mock.patch("mlonmcu.platform.lookup.get_platforms", return_value={"platform_b": MyPlatformB}):
            mlonmcu.platform.lookup.print_summary(fake_context)
            out, err = capsys.readouterr()
            split = out.split("\n")
            assert "  - platform_b" in split
            assert "  - target_1 [platform_b]" in split

    # multiple platforms, overlapping targets
    with mock.patch("mlonmcu.platform.lookup.get_platform_names", return_value=["platform_b", "platform_c"]):
        with mock.patch(
            "mlonmcu.platform.lookup.get_platforms", return_value={"platform_b": MyPlatformB, "platform_c": MyPlatformC}
        ):
            mlonmcu.platform.lookup.print_summary(fake_context)
            out, _ = capsys.readouterr()
            split = out.split("\n")
            assert "  - platform_b" in split
            assert "  - target_1 [platform_b, platform_c]" in split
            assert "  - target_2 [platform_c]" in split
