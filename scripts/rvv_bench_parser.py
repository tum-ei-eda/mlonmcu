import argparse
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser(description="Generated HTML report for RVV bench output")
parser.add_argument("input_file")
parser.add_argument("--title", default="RVV Instruction Benchmark")
parser.add_argument("--vlen", default="?")
parser.add_argument("--info", default="")
parser.add_argument("--comment", default="")
args = parser.parse_args()

input_file = Path(args.input_file)

assert input_file.is_file()

with open(input_file, "r") as f:
    lines = f.readlines()

lines = list(map(lambda x: x.strip(), lines))
lines = [line for line in lines if len(line) > 0]

# print("lines", lines, len(lines))

group_lines = defaultdict(list)
group = None

for line in lines:
    if line.startswith("<tr>"):
        assert group is not None
        group_lines[group].append(line)
    else:
        # if group is None:
        group = line

css_content = """
.base {
	margin: 40px auto;
	max-width: 1200px;
	line-height: 1.6;
	font-size: 18px;
	color: #444;
	padding: 0 10px;
}
h1, h2, h3 { line-height: 1.2 }

.tab {
	overflow: hidden;
	border: 1px solid #ccc;
	background-color: #f0f0f0;
}
.tab button {
	float: left;
	border: none;
	outline: none;
	cursor: pointer;
	padding: 15px;
}
.tab button:hover { background-color: #ddd; }
.tab button.active { background-color: #ccc; }


.tblCont {
	height: 500px;
	overflow: auto;
	resize: vertical;
}
.tblCont thead th {
	position: sticky;
	top: 0px;
	background: #fff;
}

.tblCont td:empty::before { content: "--"; }
.tblCont tr > td, tr > th { text-align: center; white-space: nowrap; }
.tblCont tr :nth-child(1) { text-align: left; font-family: monospace;  }
.tblConts tr :nth-child(2n+1) { text-align: left; font-family: monospace;  }
.tblCont tbody tr:nth-child(odd) { background-color: #f0f0f0; }
.tblContv tbody tr :nth-child(4n + 2) { background-color: #e0e0e0; }
.tblContvf tbody tr :nth-child(7n + 2) { background-color: #e0e0e0; }
.tblContvf:not(.tblContvf-show) tbody tr :nth-child(7n + 5) { background-color: #e0e0e0; }
.tblContvf:not(.tblContvf-show) tr :nth-child(7n + 2) { display: none; }
.tblContvf:not(.tblContvf-show) tr :nth-child(7n + 3) { display: none; }
.tblContvf:not(.tblContvf-show) tr :nth-child(7n + 4) { display: none; }


.result-toggles { margin-top: 5px; text-align: center; }
.result-btn { margin: 5px; }
.result-btn-hidden { color: silver; }

.center { display: flex; justify-content: center; };
"""

html_top = f"""
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>RVV benchmark SpacemiT X60</title>
	<link rel="stylesheet" href="../base.css">
	<script type="text/javascript" src="../templates/base.js"></script>
	<style>
{css_content}
  </style>
</head>
<body class="base">

<header><h1>{args.title}</h1></header>

"""

if args.comment:
    html_top += f"Comment: <i>{args.comment}</i>"

html_bot = """</body>
</html>"""


def gen_table(group, lines):
    return (
        f"""
<div class="tab">
	<button class="tabSel active" onclick="switchTbl(event, 'tblMaa')">vl={group}</button>

	<button class="tabSel"><b>Note: VLEN={args.vlen} {args.info}</b></button>
</div>

<div class="tblCont tblContvf">

<table id="tblMaa" class="tabPage" style="width:100%;">
<thead><tr><th>instruction</th><th>e8mf8</th><th>e8mf4</th><th>e8mf2</th><th>e8m1</th><th>e8m2</th><th>e8m4</th><th>e8m8</th><th>e16mf8</th><th>e16mf4</th><th>e16mf2</th><th>e16m1</th><th>e16m2</th><th>e16m4</th><th>e16m8</th><th>e32mf8</th><th>e32mf4</th><th>e32mf2</th><th>e32m1</th><th>e32m2</th><th>e32m4</th><th>e32m8</th><th>e64mf8</th><th>e64mf4</th><th>e64mf2</th><th>e64m1</th><th>e64m2</th><th>e64m4</th><th>e64m8</th></tr></thead>
<tbody>
"""
        + "\n".join(lines)
        + """
</table>

</tbody>
</table>
</div>

"""
    )


tables_contents = [gen_table(group, lines) for group, lines in group_lines.items()]

html_content = html_top + "\n</br>\n".join(tables_contents) + html_bot

print(html_content)
