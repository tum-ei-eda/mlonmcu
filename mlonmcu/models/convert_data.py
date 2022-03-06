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
import struct


def convert(mode, val):
    data = b""
    if mode == "float":
        for f in val.split(","):
            data += struct.pack("f", float(f))
    elif mode == "hexstr":
        data = val.encode("raw_unicode_escape").decode("unicode_escape").encode("raw_unicode_escape")
    elif mode == "int8":
        for i in val.split(","):
            data += struct.pack("b", int(i))
    return data


def write_file(dest, data):
    with open(dest, "wb") as f:
        f.write(data)


def main():
    if len(sys.argv) != 4:
        print(
            "Usage:",
            sys.argv[0],
            "mode(float, hexstr, int8, image, audio)",
            "value",
            "outfile",
        )
        sys.exit(1)

    mode, val, dest = sys.argv[1], sys.argv[2], sys.argv[3]

    data = convert(mode, val)
    write_file(dest, data)


if __name__ == "__main__":
    main()
