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
"""Utility to convert various types of data into MLonMCU compatible raw binary files."""
import sys
import struct
from PIL import Image
import numpy as np


def convert(mode, val):
    """Actual convertion function.

    Arguments
    ---------
    mode : str
        The chosen mode.
    val : str
        The input data.

    Returns
    -------
    data : bytes
      Raw converted output.

    Raises
    ------
    AssertionError
        If the choses mode does not exist or value has a wrong shape.
    ValueError
        If the conversion failed.

    """
    data = b""
    assert isinstance(val, str), "Input value needs to be a string"
    assert mode in ["float", "hexstr", "int8", "bmp"], f"Unsupported mode: {mode}"

    if mode == "float":
        for f in val.split(","):
            data += struct.pack("f", float(f))
    elif mode == "hexstr":
        data = val.encode("raw_unicode_escape").decode("unicode_escape").encode("raw_unicode_escape")
    elif mode == "int8":
        for i in val.split(","):
            data += struct.pack("b", int(i))
    elif mode == "bmp":
        im = Image.open(val)
        p = np.array(im)
        assert len(p.shape) in (2, 3)  # only allow grayscale or RGB
        assert p.dtype == np.uint8  # We do not want to hande endianess at this point
        data = p.tobytes()
    return data


def write_file(dest, data):
    """Utility to save the file to disk.

    Arguments
    ---------
    dest: str
       File destination.
    data: bytes
       Raw data to export.

    """
    with open(dest, "wb") as f:
        f.write(data)


def main():
    """Main entry pint handling command line options."""
    if len(sys.argv) != 4:
        print(
            "Usage:",
            sys.argv[0],
            "mode(float, hexstr, int8, bmp)",
            "value",
            "outfile",
        )
        sys.exit(1)

    mode, val, dest = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        data = convert(mode, val)
    except ValueError:
        print("Conversion Failed")
    write_file(dest, data)


if __name__ == "__main__":
    main()
