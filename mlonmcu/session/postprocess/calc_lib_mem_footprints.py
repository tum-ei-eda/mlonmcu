import sys
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

import mlonmcu.setup.utils as utils

pd.set_option("display.max_rows", None)


def demangle_fallback(func_name: str):

    exe = shutil.which("c++filt")
    assert exe is not None, "Could not find c++filt program"
    output = utils.execute(*[exe, func_name])
    return output


def unmangle_helper(func_name: Optional[str]):
    from cpp_demangle import demangle

    if func_name is None:
        return None
    if not func_name.startswith("_Z"):
        return func_name
    try:
        return demangle(func_name)
    except ValueError:
        return demangle_fallback(func_name)
    raise ValueError(f"Could not demangle symbol: {func_name}")


def parse_elf(elf_path):
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection

    with open(elf_path, "rb") as f:
        elffile = ELFFile(f)

        section = elffile.get_section_by_name(".symtab")

        assert section, "Symbol Table not found!"
        if isinstance(section, SymbolTableSection):
            total_footprint = 0
            func_footprint = {}
            for i, sym in enumerate(section.iter_symbols()):
                ty = sym.entry["st_info"]["type"]
                if ty != "STT_FUNC":
                    continue
                func = sym.name
                sz = sym.entry["st_size"]
                func_footprint[func] = sz
                total_footprint += sz
            footprint_df = pd.DataFrame(func_footprint.items(), columns=["func", "bytes"])
            footprint_df.sort_values("bytes", inplace=True, ascending=False)
            footprint_df["rel_bytes"] = footprint_df["bytes"] / total_footprint
        return footprint_df


def analyze_linker_map_helper(mapFile):
    ret = []
    data = mapFile.toJson(humanReadable=False)
    segments = data["segments"]
    for segment in segments:
        segment_name = segment["name"]
        files = segment["files"]
        for file in files:
            filepath = file["filepath"]
            if "(" in filepath:  # TODO: use regex instead
                library, obj = filepath[:-1].split("(", 1)
            else:
                library = None
                obj = filepath
            obj_short = Path(obj).name
            library_short = Path(library).name if library is not None else library
            section_type = file["sectionType"]
            symbols = file["symbols"]
            if section_type.startswith(".text."):
                symbol_name = unmangle_helper(section_type.split(".", 2)[-1])
                new = {
                    "segment": segment_name,
                    "section": section_type,
                    "symbol": symbol_name,
                    "library": library_short,
                    "library_full": library,
                    "object": obj_short,
                    "object_full": obj,
                }
                ret.append(new)
            if ".text" not in section_type:
                continue
            for symbol in symbols:
                symbol_name = symbol["name"]
                new = {
                    "segment": segment_name,
                    "section": section_type,
                    "symbol": symbol_name,
                    "library": library_short,
                    "library_full": library,
                    "object": obj_short,
                    "object_full": obj,
                }
                ret.append(new)
    return ret


def generate_pie_data(df, x: str, y: str, topk: Optional[int] = None):
    ret = df.copy()
    ret.set_index(x, inplace=True)
    ret.sort_values(y, inplace=True, ascending=False)
    if topk is not None:
        a = ret.iloc[:topk]
        b = ret.iloc[topk:].agg(others=(y, "sum"))
        ret = pd.concat([a, b])
    ret = ret[y]
    ret = ret.reset_index()
    ret.columns = [x, y]

    return ret


def agg_library_footprint(mem_footprint_df, symbol_map_df, by: str = "library", col: str = "bytes"):
    mem_footprint_df["func_unmangled"] = mem_footprint_df["func"].apply(unmangle_helper)
    ret = mem_footprint_df.set_index("func_unmangled").join(symbol_map_df.set_index("symbol"), how="left")
    ret = ret[[by, col]]
    ret = ret.groupby(by, as_index=False, dropna=False).sum()
    return ret


def main():
    assert len(sys.argv) == 3

    elf_file = Path(sys.argv[1])
    linker_map_file = Path(sys.argv[2])

    assert elf_file.is_file()
    assert linker_map_file.is_file()

    mem_footprint_df = parse_elf(elf_file)

    from mapfile_parser import mapfile

    mapFile = mapfile.MapFile()
    mapFile.readMapFile(linker_map_file)

    symbol_map = analyze_linker_map_helper(mapFile)
    symbol_map_df = pd.DataFrame(
        symbol_map,
        columns=[
            "segment",
            "section",
            "symbol",
            "object",
            "object_full",
            "library",
            "library_full",
        ],
    )

    topk = None
    mem_footprint_per_func_data = generate_pie_data(mem_footprint_df, x="func", y="bytes", topk=topk)
    library_footprint_df = agg_library_footprint(mem_footprint_df, symbol_map_df, by="library", col="bytes")
    mem_footprint_per_library_data = generate_pie_data(library_footprint_df, x="library", y="bytes", topk=topk)
    object_footprint_df = agg_library_footprint(mem_footprint_df, symbol_map_df, by="object", col="bytes")
    mem_footprint_per_object_data = generate_pie_data(object_footprint_df, x="object", y="bytes", topk=topk)
    print("per_func\n", mem_footprint_per_func_data, mem_footprint_per_func_data["bytes"].sum())
    print("per_library\n", mem_footprint_per_library_data, mem_footprint_per_library_data["bytes"].sum())
    print("per_object\n", mem_footprint_per_object_data, mem_footprint_per_object_data["bytes"].sum())


if __name__ == "__main__":
    main()
