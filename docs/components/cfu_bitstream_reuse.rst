CFU Playground bitstream reuse
============================

Keep the hardware settings and CFU source from the original hardware experiment.
For example, from the MLonMCU checkout, with the CFU workspace activated::

    mlonmcu flow run aww --target cfu_full_fpga --platform cfu_playground --backend tvmaot \
      -c cfu_playground.reuse_bitstream=1 \
      -c cfu_playground.bitstream_file=/work/git/mlonmcu/mlonmcu/workspace_cfu/temp/sessions/115/runs/24/cfu_playground/out/gateware/digilent_arty.bit \
      -c cfu_playground.wait_for_user=0 \
      -c cfu_full_fpga.fpga_tty=/dev/ttyUSB0

Add the original experiment's CPU variant, board variant, clock, RAM, UART and
``cfu_playground.verilog_file`` settings to this command. The filename in this
example does not imply that its hardware matches the default target settings.
``bitstream_file`` also accepts a source run directory or its ``cfu_playground``
project directory. It requires ``reuse_bitstream=1`` and a physical FPGA target;
combining reuse with ``use_sw_dir`` or ``use_gateware_dir`` is rejected.

Software still runs the normal ``software`` Make target. This generates LiteX
headers, linker scripts, CSR definitions, memory maps and BIOS without the
``--build`` implementation option. Reuse skips the ``bitstream`` target and
programs the run-local copy through ``common_soc.py --load-only``. This requires
the programming-only loader provided by the locally used CFU Playground fork;
older forks without that option fail with a descriptive error. No shared
Makefile or dependency checkout is patched. The existing serial file lock spans
programming, upload and monitoring; it serializes all boards on the same host.
Explicitly configured project directories must still be unique per concurrent run.

New hardware builds write ``out/gateware/bitstream_metadata.json`` and export
that file as an artifact. It records target hardware settings, the CFU SHA-256,
bitstream SHA-256, generated headers/linker scripts, CSR JSON and BIOS hashes.
Schema version 2 retains original raw SHA-256 hashes in ``generated`` and records
``raw_sha256``, ``compatibility_sha256`` and a versioned ``normalization`` method
for each artifact in ``generated_compatibility``. Raw hashes describe build
reproducibility; normalized hashes describe compatibility under these narrow rules:

* CSR JSON uses canonical key ordering and formatting, preserving every value.
* Generated C headers normalize only the timestamp in the recognized LiteX
  auto-generated comment; the LiteX revision and hardware definitions remain checked.
* ``variables.mak`` normalizes only ``BUILDINC_DIRECTORY``. CPU flags and other
  settings remain checked. Linker scripts are compared byte for byte.
* BIOS binaries normalize the recognized ``BIOS built on ...`` timestamp.
  A trailing CRC32 is recomputed only when it validates against the original
  payload, using its detected byte order. Arbitrary checksum bytes are preserved.

For the investigated sessions 128 and 135, both BIOS images were 28,560 bytes.
The only byte differences were the timestamp and the four-byte little-endian
CRC32 trailer. After timestamp normalization and validated CRC32 recomputation,
the images match. Normalization operates in memory and never edits cached files.

Reuse compares available settings before compilation and generated metadata
after compilation, and checks again before programming. CSR addresses, memory
layouts, hardware definitions, CFU RTL, CPU, clock, RAM, board and UART settings
retain hard mismatch errors. Unexplained BIOS differences produce a warning by
default because the resident BIOS may still be compatible; enable
``cfu_playground.strict_bios_compatibility=1`` to reject them. A missing BIOS
artifact remains an error. This warning does not establish bootloader compatibility.

Version 1 manifests remain supported. When original generated files are present,
reuse verifies their recorded raw hashes and computes normalized hashes directly
from those files. Source runs without manifests also use their original artifacts.
Without original files or versioned compatibility hashes, only raw comparison is
possible; differing headers/maps fail, and differing BIOS images follow the BIOS
warning/strict policy. Unknown normalization versions cannot establish a match.

A bare bitstream cannot be fully validated. Older source runs provide partial
checks using their CFU, Makefile options and generated LiteX files. Missing or
incomplete manifests produce a warning and ``partial-user-responsibility``
provenance. Board/device identity, defaults, UART and bootloader compatibility
cannot be established for arbitrary files without trusted accompanying metadata.
Even a manifest establishes consistency with recorded build inputs, not an
independent inspection of the FPGA configuration. Preserve the complete source
project and use the same CFU Playground/toolchain revisions. External HDL includes
and unrecorded changes to the SoC generator remain the user's responsibility.

Vivado reports and metrics
-------------------------

Keep reports beside the source bitstream (``*_timing.rpt``,
``*_utilization_place.rpt``, ``*_utilization_synth.rpt``, ``*_power.rpt`` and
``*_utilization_hierarchical_place.rpt``). Reuse automatically exports those
reports with the existing artifact flags. Exported ``vivado_report_*.rpt`` files
in the source run directory are a fallback. Use the existing
``--post analyse_vivado_reports`` option to obtain their metrics. Missing reports
are omitted; the postprocessor already handles an empty report set.

The reuse manifest records each report's source path and SHA-256. Original
bitstreams and reports are read only. ``FPGA Bitstream Time [s]`` is zero and
``FPGA Bitstream Reused`` is true; programming and monitoring timings retain
their existing meanings. Artifacts include the source bitstream path and hash.
