"""Parse RVV bench instruction output and render a standalone HTML report."""

import argparse
from collections import OrderedDict
from html import escape, unescape
import math
from pathlib import Path
import re

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.config import cfg, str2bool

from .postprocess import RunPostprocess


def parse_rvv_bench(text):
    """Extract complete benchmark rows, ignoring serial/ISS log messages.

    Return groups of decoded cells rather than trusting HTML from the log.
    Timing rows have two cells; vector groups start with ``vl=...`` and
    other instruction groups can use ``# Benchmark Group: ...``.
    """
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    groups = OrderedDict()
    group = "Instructions"
    pattern = (
        r"(?m)^\s*(?P<vector>vl=[^\r\n]+)|^# Benchmark Group: (?P<group>[^\r\n]+)"
        r"|<tr>\s*(?P<row>(?:<td>.*?</td>\s*)+)</tr>"
    )
    for match in re.finditer(pattern, text):
        if match["vector"] or match["group"]:
            group = (match["vector"] or match["group"]).strip()
            continue
        cells = [unescape(cell) for cell in re.findall(r"<td>(.*?)</td>", match["row"])]
        if len(cells) < 2 or not cells[0]:
            continue
        try:
            if any(value and (not math.isfinite(float(value)) or float(value) < 0) for value in cells[1:]):
                continue
        except ValueError:
            continue
        groups.setdefault(group, []).append(cells)
    if not groups:
        raise ValueError("No complete RVV bench instruction rows found in the log")
    return groups


def render_rvv_bench(text, *, title="RVV Instruction Benchmark", scalar=None, vlen="?", info="", comment=""):
    """Render scalar or vector results with embedded styles and no external assets."""
    groups = parse_rvv_bench(text)
    if scalar is None:
        scalar = all(len(cells) == 2 for rows in groups.values() for cells in rows)
    sections = []
    vector_headers = [
        f"e{sew}{lmul}" for sew in (8, 16, 32, 64) for lmul in ("mf8", "mf4", "mf2", "m1", "m2", "m4", "m8")
    ]
    for group, rows in groups.items():
        width = max(len(row) for row in rows) - 1
        if scalar:
            if width != 1:
                raise ValueError("Scalar instruction rows must contain exactly one timing value")
            unit = re.search(r"(?m)^# Benchmark Unit: ([^\r\n]+)", text)
            headers = [unit[1] if unit else "Cycles / instruction"]
        else:
            headers = vector_headers if width == len(vector_headers) else [f"Result {i + 1}" for i in range(width)]
        head = "".join(f'<th scope="col">{escape(label)}</th>' for label in ["Instruction", *headers])
        body = "\n".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
        sections.append(
            f'<section><h2>{escape(group)}</h2><div class="table"><table>'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>"
        )
    benchmark_info = re.search(r"(?m)^# Benchmark Info: ([^\r\n]+)", text)
    notes = [benchmark_info[1] if benchmark_info else "", info, comment]
    if not scalar:
        notes.insert(0, f"VLEN={vlen}")
    metadata = "".join(f"<p>{escape(str(note))}</p>" for note in notes if note)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>
body {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; font: 17px/1.5 system-ui, sans-serif; color: #222; }}
.table {{ overflow: auto; max-height: 75vh; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: .4rem .8rem; text-align: right; white-space: nowrap; }}
th {{ position: sticky; top: 0; background: #e5ebf1; }}
td:first-child, th:first-child {{ text-align: left; }}
td:first-child {{ font-family: monospace; }}
tbody tr:nth-child(even) {{ background: #f3f5f7; }}
td:empty::after {{ content: "—"; }}
</style></head><body><h1>{escape(title)}</h1>{metadata}
{''.join(sections)}
</body></html>
"""


class RVVBenchPostprocess(RunPostprocess):
    """Generate rvv_bench.html from scalar/vector instruction benchmark stdout."""

    title = cfg("RVV Instruction Benchmark")
    scalar = cfg(None, cast=str2bool)
    vlen = cfg("?")
    info = cfg("")
    comment = cfg("")

    def __init__(self, features=None, config=None):
        super().__init__("rvv_bench", features=features, config=config)

    def post_run(self, report, artifacts):
        # Runtime logs follow compile logs in the artifact chain. Select the
        # last matching log, avoiding duplicate results from build stdout.
        for artifact in reversed(artifacts):
            if not artifact.name.endswith("_out.log"):
                continue
            text = artifact.content
            if text is None and artifact.path is not None:
                text = Path(artifact.path).read_text(encoding="utf-8", errors="replace")
            if not text or "<tr>" not in text:
                continue
            html = render_rvv_bench(
                text, title=self.title, scalar=self.scalar, vlen=self.vlen, info=self.info, comment=self.comment
            )
            return [Artifact("rvv_bench.html", content=html, fmt=ArtifactFormat.TEXT, flags=("rvv_bench", "report"))]
        raise ValueError("rvv_bench postprocess requires an instruction benchmark runtime log with <tr> rows")


def main(argv=None):
    """Compatibility CLI: print HTML to stdout, or write it with --output."""
    parser = argparse.ArgumentParser(description="Generate HTML report for RVV bench output")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--title", default="RVV Instruction Benchmark")
    parser.add_argument("--vlen", default="?")
    parser.add_argument("--info", default="")
    parser.add_argument("--comment", default="")
    parser.add_argument("--scalar", action="store_const", const=True, default=None)
    args = parser.parse_args(argv)
    try:
        html = render_rvv_bench(
            args.input_file.read_text(encoding="utf-8", errors="replace"),
            title=args.title,
            scalar=args.scalar,
            vlen=args.vlen,
            info=args.info,
            comment=args.comment,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.write_text(html, encoding="utf-8")
    else:
        print(html)


if __name__ == "__main__":
    main()
