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
