"""Read-only prebuilt hardware inputs and run-local hardware provenance."""

import hashlib
import json
import re
import zlib
from pathlib import Path

from mlonmcu.logging import get_logger

logger = get_logger()
MANIFEST = "bitstream_metadata.json"
FIELDS = ("fpga_target", "fpga_variant", "cpu_variant", "sys_clk_freq", "integrated_main_ram_size", "baud")


def sha256(path):
    with open(path, "rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def hardware_config(platform, target):
    result = {key: getattr(target, key) for key in FIELDS}
    cfu = (platform.project_dir / "cfu.v") if platform.project_dir else Path("/nonexistent-cfu")
    if platform.project_dir and (platform.project_dir / "cfu.sv").is_file():
        cfu = platform.project_dir / "cfu.sv"
    if cfu.is_file():
        result["cfu_sha256"] = sha256(cfu)
    return result


def resolve(platform, target):
    """Validate before software compilation; never write into the source run."""
    if not platform.reuse_bitstream:
        if platform.bitstream_file is not None:
            raise ValueError("cfu_playground.bitstream_file requires reuse_bitstream=1")
        return None
    if not target.fpga_sim or target.rtl_sim:
        raise ValueError("Bitstream reuse requires an FPGA target")
    if target.use_sw_dir or target.use_gateware_dir:
        raise ValueError("Bitstream reuse cannot be combined with use_sw_dir/use_gateware_dir")
    if platform.bitstream_file is None:
        raise ValueError("reuse_bitstream requires cfu_playground.bitstream_file")
    path = platform.bitstream_file.expanduser().resolve()
    if path.is_dir():
        project = path / "cfu_playground" if (path / "cfu_playground").is_dir() else path
        path = project / "out/gateware" / f"{target.fpga_target}.bit"
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Prebuilt bitstream is missing or empty: {path}")
    if platform.project_dir and platform.project_dir.resolve() in path.parents:
        raise ValueError("Prebuilt bitstream must be outside the current run directory")
    project = path.parent.parent.parent
    metadata_path = path.parent / MANIFEST
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    digest = sha256(path)
    if metadata and metadata.get("bitstream_sha256") != digest:
        raise ValueError(f"Bitstream SHA-256 does not match metadata: {path}")
    expected = metadata.get("hardware", {}).copy()
    # Older runs have no manifest, but retain the CFU and exported LiteX options.
    cfu = project / "cfu.sv" if (project / "cfu.sv").is_file() else project / "cfu.v"
    if cfu.is_file():
        expected.setdefault("cfu_sha256", sha256(cfu))
    makefile = project / "Makefile"
    if makefile.is_file():
        text = makefile.read_text()
        for key, option in (
            ("fpga_variant", "variant"),
            ("cpu_variant", "cpu-variant"),
            ("sys_clk_freq", "sys-clk-freq"),
            ("integrated_main_ram_size", "integrated-main-ram-size"),
        ):
            match = re.search(r"--" + option + r"[= ]([^\s]+)", text)
            if match:
                value = match[1]
                if key in ("sys_clk_freq", "integrated_main_ram_size"):
                    value = int(float(value))
                expected.setdefault(key, value)
    if target.fpga_target == "digilent_arty":
        reports = list(path.parent.glob("*.rpt"))
        reports += list(project.parent.glob("vivado_report_*.rpt"))
        for report in reports:
            device = re.search(r"Device\s*:\s*(?:xc)?7a(35|100)t", report.read_text(errors="replace"), re.I)
            variant = re.fullmatch(r"a7-(35|100)", target.fpga_variant)
            if device and variant and device[1] != variant[1]:
                raise ValueError(f"Prebuilt hardware mismatch for FPGA device in {report}")
    current = hardware_config(platform, target)
    for key, value in expected.items():
        if key in current and value != current[key]:
            raise ValueError(f"Prebuilt hardware mismatch for {key}: source={value!r}, requested={current[key]!r}")
    # Require the explicit programming-only implementation inspected by this platform.
    loader = platform.cfu_playground_src_dir / "soc/common_soc.py"
    if "--load-only" not in loader.read_text():
        raise ValueError("Bitstream reuse requires CFU Playground common_soc.py with --load-only support")
    complete = (
        metadata.get("compatibility") != "partial-user-responsibility"
        and all(key in metadata.get("hardware", {}) for key in (*FIELDS, "cfu_sha256"))
        and "csr.json" in metadata.get("generated", {})
        and "software/bios/bios.bin" in metadata.get("generated", {})
    )
    if not complete:
        logger.warning(
            "Prebuilt bitstream has incomplete hardware metadata; compatibility is user responsibility: %s", path
        )
    return {
        "source": str(path),
        "bitstream_sha256": digest,
        "hardware": current,
        "compatibility": "manifest-and-generated-metadata" if complete else "partial-user-responsibility",
        "project": str(project),
        "source_metadata": metadata,
    }


NORMALIZATION_VERSION = "litex-v1"
BIOS = "software/bios/bios.bin"
BIOS_TIMESTAMP = re.compile(
    rb"(BIOS built on )[A-Z][a-z]{2} [ 0-9][0-9] [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}(?=\x00|\r|\n|$)"
)


def artifact_hashes(path, name):
    """Hash original bytes and a narrowly normalized, in-memory compatibility view."""
    data = path.read_bytes()
    normalized = data
    method = "identity"
    if name == "csr.json":
        # Keep every value (including addresses and memory sizes).
        normalized = json.dumps(json.loads(data), sort_keys=True, separators=(",", ":")).encode()
        method = "canonical-json"
    elif name == BIOS:
        # LiteX crcfbigen appends a CRC32. Only rewrite a verified checksum,
        # never discard arbitrary trailing bytes or mask an invalid checksum.
        endian = next(
            (
                order
                for order in ("little", "big")
                if len(data) >= 4 and int.from_bytes(data[-4:], order) == zlib.crc32(data[:-4])
            ),
            None,
        )
        payload = data[:-4] if endian else data
        normalized = BIOS_TIMESTAMP.sub(rb"\g<1><timestamp>", payload)
        method = "bios-build-timestamp"
        if endian:
            normalized += zlib.crc32(normalized).to_bytes(4, endian)
            method += "-crc32-" + endian
    elif name.startswith("software/include/generated/") and name.endswith(".h"):
        normalized = re.sub(
            rb"(?m)^(// Auto-generated by LiteX \([^\r\n)]+\) on )"
            rb"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(?=\r?$)",
            rb"\g<1><timestamp>",
            data,
        )
        method = "litex-header-timestamp"
    elif name == "software/include/generated/variables.mak":
        normalized = re.sub(rb"(?m)^(BUILDINC_DIRECTORY[ \t]*[:?]?=)[^\r\n]*", rb"\g<1><run-local>", data)
        method = "litex-buildinc-directory"
    return {
        "raw_sha256": hashlib.sha256(data).hexdigest(),
        "compatibility_sha256": hashlib.sha256(normalized).hexdigest(),
        "normalization": NORMALIZATION_VERSION + ":" + method,
    }


def check_generated(platform, state):
    """Hard-check hardware interfaces; report unexplained resident BIOS differences."""
    source = Path(state["project"]) / "out"
    metadata = state["source_metadata"]
    expected = metadata.get("generated", {}).copy()
    recorded = metadata.get("generated_compatibility", {})
    # Also cover files omitted by old manifests (notably variables.mak).
    if source.is_dir():
        for name, digest in generated_hashes(source).items():
            expected.setdefault(name, digest)
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid generated metadata path: {name}")
        current = platform.out_dir / relative
        original = source / relative
        if not current.is_file():
            raise ValueError(f"Prebuilt hardware mismatch in generated LiteX metadata: {name} (missing)")
        actual = artifact_hashes(current, name)
        reference = recorded.get(name)
        if original.is_file():
            reference = artifact_hashes(original, name)
            if reference["raw_sha256"] != digest:
                raise ValueError(f"Source generated artifact SHA-256 does not match metadata: {name}")
        if actual["raw_sha256"] == digest:
            continue
        if (
            reference
            and reference.get("normalization") == actual["normalization"]
            and reference.get("compatibility_sha256") == actual["compatibility_sha256"]
        ):
            continue
        message = f"Prebuilt hardware mismatch in generated LiteX metadata: {name}"
        if name == BIOS:
            message += " (BIOS differs beyond recognized normalization or only a historical raw hash is available)"
            if not platform.strict_bios_compatibility:
                logger.warning("%s; resident BIOS compatibility is user responsibility", message)
                continue
        raise ValueError(message)


def generated_paths(out_dir):
    paths = [out_dir / "csr.json", out_dir / BIOS]
    include = out_dir / "software/include/generated"
    if include.is_dir():
        paths.extend(
            p for p in include.rglob("*") if p.is_file() and (p.suffix in (".h", ".ld") or p.name == "variables.mak")
        )
    return sorted(p for p in paths if p.is_file())


def generated_hashes(out_dir):
    # Retain the v1 raw-hash mapping for older consumers.
    return {str(p.relative_to(out_dir)): sha256(p) for p in generated_paths(out_dir)}


def generated_compatibility(out_dir):
    return {
        str(p.relative_to(out_dir)): artifact_hashes(p, str(p.relative_to(out_dir))) for p in generated_paths(out_dir)
    }


def provenance(platform, target, path, state=None):
    result = {
        "schema_version": 2,
        "bitstream_sha256": sha256(path),
        "hardware": hardware_config(platform, target),
        "generated": generated_hashes(platform.out_dir),
        "generated_compatibility": generated_compatibility(platform.out_dir),
        "reused": state is not None,
    }
    if state:
        result.update({key: state[key] for key in ("source", "compatibility")})
    return result
