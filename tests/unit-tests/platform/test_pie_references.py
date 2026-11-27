"""Exercise the C reference calculations independently of PIE hardware."""

from pathlib import Path
import shutil
import subprocess

import pytest

from mlonmcu.platform.espidf.espidf import get_project_template


def test_pie_accumulator_references(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler is not installed")
    source = Path(get_project_template("rvv_bench")) / "main" / "pie_bench.c"
    harness = tmp_path / "references.c"
    harness.write_text(
        f'#include "{source}"\n'
        "#include <assert.h>\n"
        "#define PIE_CASE(ID, OP, GROUP, LABEL, CAPTURE, SAR, ...) \\\n"
        "void pie_check_##ID(const void *a, const void *b, void *c) {} \\\n"
        "uint32_t pie_bench_##ID(const void *a, const void *b, void *c) { return 0; }\n"
        '#include "pie_cases.def"\n'
        "int main(void) {\n"
        "assert(expected(MUL, 0) == 0x7fff);\n"
        "assert(expected(NARROW_MUL16, 0) == 0x8000);\n"
        "assert(expected(MUL8, 0) == 0x7f00);\n"
        "assert(expected_dot(DOT16) == -59934925);\n"
        "uint32_t negative[2] = {0xfc6d7733U, 0xffU};\n"
        "assert(decode_xacc(negative) == -59934925);\n"
        "uint32_t positive[2] = {0x12345678U, 1U};\n"
        "assert(decode_xacc(positive) == INT64_C(0x112345678));\n"
        "return 0;\n}\n"
    )
    executable = tmp_path / "references"
    subprocess.run([cc, "-std=c99", "-I", str(source.parent), str(harness), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
