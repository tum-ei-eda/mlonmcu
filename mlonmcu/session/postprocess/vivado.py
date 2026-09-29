"""Utilities for extracting metrics from Vivado text report artifacts."""

import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd

NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def _read(path):
    return path.read_text(errors="replace") if path and path.is_file() else ""


def _to_num(value):
    value = value.strip().replace(",", "")
    if value in {"", "-", "NA", "N/A"}:
        return None
    try:
        value = float(value)
        return int(value) if value.is_integer() else value
    except ValueError:
        return value


def _find_report(directory, suffix):
    matches = sorted(directory.glob(f"*_{suffix}.rpt"))
    return matches[0] if matches else None


def _metadata(text):
    patterns = {
        "vivado_version": r"\| Tool Version\s*:\s*(.+?)\s*$",
        "report_date": r"\| Date\s*:\s*(.+?)\s*$",
        "host": r"\| Host\s*:\s*(.+?)\s*$",
        "design": r"\| Design\s*:\s*(.+?)\s*$",
        "device": r"\| Device\s*:\s*(.+?)\s*$",
        "design_state": r"\| Design State\s*:\s*(.+?)\s*$",
    }
    return {
        key: match.group(1).strip() for key, pattern in patterns.items() if (match := re.search(pattern, text, re.M))
    }


def _table_row(text, label):
    match = re.search(r"^\|\s*" + re.escape(label) + r"\*?\s*\|(.+)$", text, re.M)
    return [item.strip() for item in match.group(1).split("|")[:-1]] if match else None


def _parse_utilization(text, prefix):
    labels = {
        "Slice LUTs": "slice_luts",
        "LUT as Logic": "lut_as_logic",
        "LUT as Memory": "lut_as_memory",
        "Slice Registers": "slice_registers",
        "Slice": "slices",
        "Block RAM Tile": "bram_tiles",
        "RAMB36/FIFO*": "ramb36_fifo",
        "RAMB18": "ramb18",
        "DSPs": "dsps",
        "Bonded IOB": "bonded_iob",
        "BUFGCTRL": "bufgctrl",
        "PLLE2_ADV": "plle2_adv",
    }
    result = {}
    for label, stem in labels.items():
        values = _table_row(text, label)
        if values and len(values) >= 5:
            result.update(
                {
                    f"{prefix}{stem}_used": _to_num(values[0]),
                    f"{prefix}{stem}_available": _to_num(values[3]),
                    f"{prefix}{stem}_util_pct": _to_num(values[4]),
                }
            )
    return result


def _parse_hierarchical_utilization(text):
    wanted = {"VexRiscv": "cpu", "IBusCachedPlugin_cache": "icache", "dataCache_1": "dcache", "Cfu": "cfu"}
    row_re = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|" + r"\s*([0-9.]+)\s*\|" * 8 + r"$", re.M)
    result = {}
    keys = ["total_luts", "logic_luts", "lutram", "srl", "ffs", "ramb36", "ramb18", "dsps"]
    for match in row_re.finditer(text):
        prefix = wanted.get(match.group(1).strip())
        if prefix:
            values = [float(value) for value in match.groups()[2:]]
            result.update(
                {
                    f"hier_{prefix}_{key}": int(value) if value.is_integer() else value
                    for key, value in zip(keys, values)
                }
            )
            result[f"hier_{prefix}_bram_tiles"] = values[5] + values[6] / 2
    return result


def _parse_timing(text):
    result = {"timing_met": "All user specified timing constraints are met." in text}
    marker = text.find("| Design Timing Summary")
    if marker >= 0:
        lines = text[marker : marker + 2500].splitlines()
        for index, line in enumerate(lines):
            if "WNS(ns)" in line and "TNS(ns)" in line:
                for candidate in lines[index + 1 : index + 6]:
                    values = re.findall(NUM_RE, candidate)
                    if len(values) >= 12:
                        keys = [
                            "wns_ns",
                            "tns_ns",
                            "tns_failing_endpoints",
                            "tns_total_endpoints",
                            "whs_ns",
                            "ths_ns",
                            "ths_failing_endpoints",
                            "ths_total_endpoints",
                            "wpws_ns",
                            "tpws_ns",
                            "tpws_failing_endpoints",
                            "tpws_total_endpoints",
                        ]
                        result.update(dict(zip(keys, map(_to_num, values[:12]))))
                        break
                break
    return result


def _parse_clock_summary(text):
    marker = text.find("| Clock Summary")
    if marker < 0:
        return pd.DataFrame(columns=["clock", "period_ns", "frequency_mhz"])
    rows = []
    for line in text[marker : marker + 5000].splitlines():
        match = re.match(rf"^\s*([A-Za-z0-9_./\[\]-]+)\s+\{{[^}}]+\}}\s+({NUM_RE})\s+({NUM_RE})\s*$", line)
        if match:
            rows.append(
                {
                    "clock": match.group(1),
                    "period_ns": float(match.group(2)),
                    "frequency_mhz": float(match.group(3)),
                }
            )
    return pd.DataFrame(rows)


def _parse_intra_clock_wns(text):
    marker = text.find("| Intra Clock Table")
    if marker < 0:
        return {}
    result = {}
    for line in text[marker : marker + 7000].splitlines():
        match = re.match(rf"^\s*([A-Za-z0-9_./\[\]-]+)\s+({NUM_RE})\s+({NUM_RE})\s+", line)
        if match:
            result[match.group(1)] = float(match.group(2))
    return result


def _choose_clock(clocks, requested):
    if clocks.empty:
        return None
    for candidate in (requested, "soc_crg_clkout0", "sys_clk", "clk100"):
        selected = clocks[clocks["clock"] == candidate]
        if not selected.empty:
            return selected.iloc[0]
    return clocks.iloc[0]


def _parse_power(text):
    labels = {
        "Total On-Chip Power (W)": "power_total_w",
        "Dynamic (W)": "power_dynamic_w",
        "Device Static (W)": "power_static_w",
        "Junction Temperature (C)": "junction_temp_c",
        "Max Ambient (C)": "max_ambient_c",
        "Confidence Level": "power_confidence",
    }
    result = {}
    for label, key in labels.items():
        match = re.search(r"^\|\s*" + re.escape(label) + r"\s*\|\s*([^|]+?)\s*\|", text, re.M)
        if match:
            result[key] = _to_num(match.group(1))
    return result


def parse_gateware_dir(directory, clock="soc_crg_clkout0"):
    """Parse the reports in one Vivado gateware directory into one record."""
    directory = Path(directory)
    place = _read(_find_report(directory, "utilization_place"))
    synth = _read(_find_report(directory, "utilization_synth"))
    timing = _read(_find_report(directory, "timing"))
    power = _read(_find_report(directory, "power"))
    hierarchical = _read(_find_report(directory, "utilization_hierarchical_place"))
    result = {"name": directory.name}
    result.update(_metadata(place or timing))
    result.update(_parse_utilization(synth, "synth_"))
    result.update(_parse_utilization(place, "place_"))
    result.update(_parse_hierarchical_utilization(hierarchical))
    result.update(_parse_timing(timing))
    result.update(_parse_power(power))
    selected_clock = _choose_clock(_parse_clock_summary(timing), clock)
    if selected_clock is not None:
        result["clock"] = selected_clock.clock
        result["clock_period_ns"] = selected_clock.period_ns
        result["clock_freq_mhz"] = selected_clock.frequency_mhz
        wns = _parse_intra_clock_wns(timing).get(selected_clock.clock)
        if wns is not None and selected_clock.period_ns > wns:
            result["clock_wns_ns"] = wns
            result["estimated_min_period_ns"] = selected_clock.period_ns - wns
            result["estimated_fmax_mhz"] = 1000.0 / result["estimated_min_period_ns"]
    return result


def parse_vivado_artifacts(artifacts, clock="soc_crg_clkout0"):
    """Parse individual Vivado report artifacts as one temporary gateware directory."""
    with tempfile.TemporaryDirectory() as tempdir:
        gateware_dir = Path(tempdir) / "gateware"
        gateware_dir.mkdir()
        for artifact in artifacts:
            report_type = next(
                (
                    flag
                    for flag in artifact.flags
                    if flag
                    in {"utilization_place", "utilization_synth", "timing", "power", "utilization_hierarchical_place"}
                ),
                None,
            )
            if report_type is None:
                continue
            destination = gateware_dir / f"report_{report_type}.rpt"
            if artifact.content is not None:
                destination.write_text(artifact.content)
            elif artifact.path is not None:
                shutil.copyfile(artifact.path, destination)
        return pd.DataFrame([parse_gateware_dir(gateware_dir, clock=clock)])
