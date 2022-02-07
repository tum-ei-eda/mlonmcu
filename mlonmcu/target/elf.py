#!/usr/bin/env python

# import os
# import sys
import csv
import argparse
import humanize
import elftools.elf.elffile as elffile

# from elftools.elf.constants import SH_FLAGS
from elftools.elf.sections import SymbolTableSection

from mlonmcu.logging import get_logger

logger = get_logger()


"""
Script to gather metrics on static ROM and RAM usage.

Heavility inspired by get_metrics.py found in the ETISS repository
"""


def parseElf(inFile):
    # TODO: check if this is generic anough for multiple platforms (riscv, arm, x86)
    # TODO: comare results with `riscv32-unknown-elf-size`
    m = {}
    m["rom_rodata"] = 0
    m["rom_code"] = 0
    m["rom_misc"] = 0
    m["ram_data"] = 0
    m["ram_zdata"] = 0
    heapStart = None

    ignoreSections = [
        "",
        ".stack",
        ".comment",
        ".riscv.attributes",
        ".strtab",
        ".shstrtab",
    ]

    with open(inFile, "rb") as f:
        e = elffile.ELFFile(f)

        for s in e.iter_sections():
            if s.name.startswith(".text"):
                m["rom_code"] += s.data_size
            elif s.name.startswith(".srodata"):
                m["rom_rodata"] += s.data_size
            elif s.name.startswith(".sdata"):
                m["ram_data"] += s.data_size
            elif s.name == ".rodata":
                m["rom_rodata"] += s.data_size
            elif s.name == ".vectors" or s.name == ".init_array":
                m["rom_misc"] += s.data_size
            elif s.name == ".data":
                m["ram_data"] += s.data_size
            elif s.name == ".bss" or s.name == ".sbss" or s.name == ".shbss":
                m["ram_zdata"] += s.data_size
            elif s.name.startswith(".gcc_except"):
                pass
            elif s.name.startswith(".sdata2"):
                pass
            elif s.name.startswith(".debug_"):
                pass
            elif s.name in ignoreSections:
                pass
            else:
                logger.debug("ignored: " + s.name + " / size: " + str(s.data_size))

    return m


def printSz(sz, unknown_msg=""):
    if sz is None:
        return f"unknown [{unknown_msg}]" if unknown_msg else "unknown"
    return humanize.naturalsize(sz) + " (" + hex(sz) + ")"


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "elf", metavar="ELF", type=str, nargs=1, help="The target ELF file"
    )
    parser.add_argument(
        "--out",
        "-o",
        metavar="FILE",
        type=str,
        default="",
        help="""Output CSV file (default: -)""",
    )
    args = parser.parse_args()

    elfFile = args.elf[0]
    csvFile = args.out

    return elfFile, csvFile


def get_results(elfFile):
    staticSizes = parseElf(elfFile)

    romSize = sum([staticSizes[k] for k in staticSizes if k.startswith("rom_")])
    ramSize = sum([staticSizes[k] for k in staticSizes if k.startswith("ram_")])

    results = {
        "rom": romSize,
        "rom_rodata": staticSizes["rom_rodata"],
        "rom_code": staticSizes["rom_code"],
        "rom_misc": staticSizes["rom_misc"],
        "ram": ramSize,
        "ram_data": staticSizes["ram_data"],
        "ram_zdata": staticSizes["ram_zdata"],
    }
    return results


def print_results(results):
    print("=== Results ===")
    print("ROM usage:        " + printSz(results["rom"]))
    print("  read-only data: " + printSz(results["rom_rodata"]))
    print("  code:           " + printSz(results["rom_code"]))
    print("  other required: " + printSz(results["rom_misc"]))
    print("RAM usage:        " + printSz(results["ram"]))
    print("  data:           " + printSz(results["ram_data"]))
    print("  zero-init data: " + printSz(results["ram_zdata"]))


def write_csv(filename, results):
    # Write metrics to file
    if csvFile:
        with open(csvFile, "w") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)


def main():
    elfFile, csvFile = parse_cmdline()

    results = get_results(elfFile)

    print_results(results)

    if csvFile:
        write_csv(csvFile, results)


if __name__ == "__main__":
    main()
