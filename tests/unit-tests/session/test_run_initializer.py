#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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

import yaml

from mlonmcu.session.run import RunInitializer


def test_load_multiple_run_initializers(tmp_path):
    path = tmp_path / "initializer.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {"model_name": "model_a", "target_name": "etiss_rv32"},
                    {"model_name": "model_b", "target_name": "spike"},
                ]
            }
        ),
        encoding="utf-8",
    )

    initializers = RunInitializer.from_file(path)

    assert [initializer.model_name for initializer in initializers] == ["model_a", "model_b"]
    assert [initializer.target_name for initializer in initializers] == ["etiss_rv32", "spike"]
    assert all(initializer.frozen for initializer in initializers)
