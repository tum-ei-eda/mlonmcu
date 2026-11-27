"""Execute MLIF firmware on generated SCAIE-V/Nailgun CVA5 or Vex RTL."""

import json
from pathlib import Path
import subprocess
import sys

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.config import str2bool
from .riscv import RISCVTarget


class NailgunTarget(RISCVTarget):
    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "arch": "rv32im_zicsr_zifencei",
        "abi": "ilp32",
        "fpu": "none",
        "compressed": False,
        "atomic": False,
        "core": "cva5",
        "rtl_dir": None,
        "runner": None,
        "execution": "rtl",
        "fpga_runner": None,
        "fpga_vivado": "/nfs/tools/xilinx/Vivado/2024.1/bin/vivado",
        "fpga_board_serial": "210319C08AE1",
        "fpga_port": "/dev/serial/by-id/usb-Digilent_Digilent_USB_Device_210319C08AE1-if01-port0",
        "fpga_uart_container": None,
        "verilator": "verilator",
        "max_cycles": 100000000,
        "timeout": 600,
        "jobs": 4,
        "data_wait": 0,
        "ram_size": 262144,
        "stack_size": 8192,
        "heap_size": 4096,
        "bench_dag": True,
    }

    def __init__(self, name="nailgun", features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        if self.xlen != 32 or self.abi != "ilp32":
            raise ValueError("Nailgun simulation requires RV32 / ILP32")

    def get_target_system(self):
        return "nailgun"

    def get_platform_defs(self, platform):
        defs = super().get_platform_defs(platform)
        defs.update(
            {
                "NAILGUN_RAM_SIZE": int(self.config["ram_size"]),
                "NAILGUN_STACK_SIZE": int(self.config["stack_size"]),
                "NAILGUN_HEAP_SIZE": int(self.config["heap_size"]),
                "GLOBAL_ISEL_BENCH_DAG": str2bool(self.config["bench_dag"]),
            }
        )
        return defs

    def exec(self, program, *args, cwd=None, **kwargs):
        if kwargs.get("stdin_data"):
            raise ValueError("Nailgun supports ROM inputs; set mlif.set_inputs_interface=rom")
        if args:
            raise ValueError("Nailgun firmware does not accept command-line input")
        mode = self.config["execution"]
        if mode not in ("rtl", "fpga"):
            raise ValueError("nailgun.execution must be rtl or fpga")
        runner = self.config["fpga_runner"] if mode == "fpga" else self.config["runner"]
        if not runner or not self.config["rtl_dir"]:
            raise ValueError("Set nailgun.rtl_dir and the selected execution runner")
        directory = Path(cwd or ".").resolve() / ("nailgun_fpga" if mode == "fpga" else "nailgun_sim")
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / "firmware.bin"
        objcopy = Path(self.riscv_gcc_prefix) / "bin" / (str(self.riscv_gcc_basename) + "-objcopy")
        subprocess.run([str(objcopy), "-O", "binary", str(Path(program).resolve()), str(binary)], check=True)
        command = [
            sys.executable,
            str(runner),
            "--core",
            self.config["core"],
            "--rtl",
            str(self.config["rtl_dir"]),
            "--firmware",
            str(binary),
            "--output",
            str(directory),
        ]
        if mode == "fpga":
            for option in ("vivado", "board_serial", "port", "uart_container"):
                value = self.config["fpga_" + option]
                if value:
                    command += ["--" + option.replace("_", "-"), str(value)]
            options = ("timeout", "ram_size")
        else:
            command += ["--verilator", str(self.config["verilator"])]
            options = ("max_cycles", "timeout", "jobs", "data_wait", "ram_size")
        for option in options:
            command += ["--" + option.replace("_", "-"), str(self.config[option])]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        uart = directory / "uart.log"
        out = uart.read_text(errors="replace") if uart.exists() else result.stdout
        meta = directory / "execution.json"
        if not meta.exists() or result.returncode or not json.loads(meta.read_text())["passed"]:
            raise RuntimeError(f"Nailgun {mode} execution failed:\n" + out + "\n" + result.stdout)
        handler = kwargs.get("handle_exit")
        if handler is not None:
            status = handler(0, out=out)
            if status:
                raise RuntimeError(f"MLIF validation failed with status {status}")
        artifacts = [
            Artifact("nailgun_" + name, content=(directory / name).read_text(errors="replace"), fmt=ArtifactFormat.TEXT)
            for name in ("execution.json", "uart.log", "compile.log")
        ]
        artifacts.append(Artifact("nailgun_uart.bin", raw=uart.read_bytes(), fmt=ArtifactFormat.BIN))
        if mode == "fpga":
            build = directory / "synthesis"
            for name in ("timing_summary.rpt", "utilization_impl.rpt", "drc_impl.rpt"):
                artifacts.append(
                    Artifact(
                        "nailgun_fpga_" + name, content=(build / "reports" / name).read_text(), fmt=ArtifactFormat.TEXT
                    )
                )
            for source, name in ((build / "vex_arty_top.bit", "arty.bit"), (build / "routed.dcp", "routed.dcp")):
                artifacts.append(
                    Artifact("nailgun_fpga_" + name, raw=source.read_bytes(), fmt=ArtifactFormat.BIN, optional=True)
                )
        if kwargs.get("live"):
            kwargs.get("print_func", print)(out)
        if kwargs.get("encoding", "utf-8") is None:
            out = uart.read_bytes()
        return out, artifacts
