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
import sys

from .tvmaot import TVMAOTBackend, main


class TVMAOTPlusBackend(TVMAOTBackend):
    DEFAULTS = {
        **TVMAOTBackend.DEFAULTS,
        "arena_size": 0,
        "unpacked_api": True,
    }

    name = "tvmaotplus"

    def __init__(self, runtime="crt", fmt="mlf", system_lib=False, features=None, config=None):
        super().__init__(runtime=runtime, fmt=fmt, system_lib=False, features=features, config=config)

    @property
    def extra_pass_config(self):
        # Make sure that defaults are not dropped
        defaults = {
            "tir.usmp.algorithm": "hill_climb",
        }
        forced = {
            "tir.usmp.enable": True,
        }
        new = super().extra_pass_config
        ret = {
            **defaults,
            **new,
            **forced,
        }
        return ret


if __name__ == "__main__":
    sys.exit(
        main(
            TVMAOTPlusBackend,
            args=sys.argv[1:],
        )
    )  # pragma: no cover
