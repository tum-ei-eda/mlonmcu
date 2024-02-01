#!/usr/bin/env python
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
"""ELF File Tool"""

# import os
# import sys
import csv
import argparse
from elftools.elf import elffile

from mlonmcu.logging import get_logger

logger = get_logger()


"""
Script to gather metrics on static ROM and RAM usage.

Heavility inspired by get_metrics.py found in the ETISS repository
"""


def parseElf(inFile):
    """Extract static memory usage details from ELF file by mapping each segment."""
    # TODO: check if this is generic anough for multiple platforms (riscv, arm, x86)
    # TODO: comare results with `riscv32-unknown-elf-size`
    m = {}
    m["rom_rodata"] = 0
    m["rom_code"] = 0
    m["rom_misc"] = 0
    m["ram_data"] = 0
    m["ram_zdata"] = 0

    ignoreSections = [
        "",
        ".stack",
        ".comment",
        ".riscv.attributes",
        ".strtab",
        ".stabstr",
        ".shstrtab",
        ".symtab",
        ".eh_frame",
        ".stab",
        ".heap",  # ?
        # The following are x86 only:
        ".interp",
        ".dynsym",
        ".dynstr",
        ".dynamic",
        ".got",
        ".data.rel.ro",
        # Espressif
        ".flash.appdesc",
        ".iram0.text_end",  # ?
        # QEMU
        ".htif",
        # Zephyr
        ".mcuboot_header",
        ".metadata",
        "ctors",
        "initlevel",
        "devices",
        "device_handles",
        "sw_isr_table",
        "device_states",
        ".mcuboot_header",
        ".metadata",
        "ctors",
        "initlevel",
        "devices",
        "device_handles",
        "sw_isr_table",
        "device_states",
        ".xt.prop",
        ".xt.lit",
        "k_heap_area",
        "datas",
        # Pulp
        ".data_tiny_fc",
        ".data_tiny_l1",
        ".l1cluster_g",
        ".heap_l2_shared",
        ".Pulp_Chip.Info",
        # ARM (corstone300)
        ".ddr",
        # cv32e40p
        ".debugger_stack",
        # ara
        ".l2",
        # vicuna (ram)
        ".user_align",
    ]
    ignorePrefixes = [
        ".gcc_except",
        ".sdata2",
        ".debug_",
        # ARM only:
        ".ARM",
        # The following are x86 only:
        ".note",
        ".gnu",
        ".rela",
        ".plt",
    ]
    ignoreSuffixes = [
        ".table",
        "dummy",
        "heap_start",
        "rom_start",
        ".info",
    ]

    with open(inFile, "rb") as f:
        e = elffile.ELFFile(f)

        for s in e.iter_sections():
            if s.name.startswith(".text") or s.name.endswith(".text") or s.name == "text":
                m["rom_code"] += s.data_size
            elif s.name.startswith(".srodata"):
                m["rom_rodata"] += s.data_size
            elif s.name.startswith(".sdata"):
                m["ram_data"] += s.data_size
            elif s.name.endswith(".rodata") or s.name == "rodata":
                m["rom_rodata"] += s.data_size
            elif s.name in [
                ".vectors",
                "iram0.vectors",
                ".iram0.vectors",
                ".init_array",
                ".fini_array",
                ".fini",
                ".init",
                ".eh_frame",
                ".eh_frame_hdr",
            ]:
                m["rom_misc"] += s.data_size
            elif s.name.endswith(".data"):
                m["ram_data"] += s.data_size
            elif (
                s.name == ".bss"
                or s.name == "bss"
                or s.name == ".sbss"
                or s.name == ".shbss"
                or s.name == ".bss.noinit"
                or s.name.endswith(".bss")
                or s.name.startswith(".bss")
                or s.name.startswith(".sbss")
                or s.name == "noinit"
            ):
                m["ram_zdata"] += s.data_size
            elif s.name in ignoreSections:
                pass
            elif any(s.name.startswith(prefix) for prefix in ignorePrefixes):
                pass
            elif any(s.name.endswith(suffix) for suffix in ignoreSuffixes):
                pass
            elif s.data_size == 0:
                pass  # No warning for empty sections
            else:
                logger.warning("ignored: %s / size: %d", s.name, s.data_size)

    return m


def printSz(sz, unknown_msg=""):
    """Helper function for printing file sizes."""
    if sz is None:
        return f"unknown [{unknown_msg}]" if unknown_msg else "unknown"
    import humanize

    return humanize.naturalsize(sz) + " (" + hex(sz) + ")"


def parse_cmdline():
    """Cmdline interface definition."""
    parser = argparse.ArgumentParser()
    parser.add_argument("elf", metavar="ELF", type=str, nargs=1, help="The target ELF file")
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
    """Converts and returns collected data."""
    staticSizes = parseElf(elfFile)

    romSize = sum([size for key, size in staticSizes.items() if key.startswith("rom_")])
    ramSize = sum([size for key, size in staticSizes.items() if key.startswith("ram_")])

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
    """Displaying a fancy overview."""
    print("=== Results ===")
    print("ROM usage:        " + printSz(results["rom"]))
    print("  read-only data: " + printSz(results["rom_rodata"]))
    print("  code:           " + printSz(results["rom_code"]))
    print("  other required: " + printSz(results["rom_misc"]))
    print("RAM usage:        " + printSz(results["ram"]))
    print("  data:           " + printSz(results["ram_data"]))
    print("  zero-init data: " + printSz(results["ram_zdata"]))


def write_csv(filename, results):
    """Utility for writing a CSV file."""
    # Write metrics to file
    if filename:
        with open(filename, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)


def main():
    """Main entry point for command line usage."""
    elfFile, csvFile = parse_cmdline()

    results = get_results(elfFile)

    print_results(results)

    if csvFile:
        write_csv(csvFile, results)


if __name__ == "__main__":
    main()
