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
"""Record file hashes and sample count for a complete model validation dataset."""

import argparse
from pathlib import Path

from mlonmcu.models.data_utils import write_model_data_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("paths", nargs="+", help="Data directories and label files, relative to model_dir")
    parser.add_argument("--output", type=Path, help="Defaults to MODEL_DIR/data_manifest.json; must not exist")
    args = parser.parse_args()
    manifest = write_model_data_manifest(args.model_dir, args.paths, destination=args.output)
    print(f"Recorded {len(manifest['files'])} files for {manifest['sample_count']} samples")


if __name__ == "__main__":
    main()
