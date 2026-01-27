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
"""MLIF Litex Platform"""
import os
import time

# import tempfile
from typing import Tuple
from pathlib import Path

from mlonmcu.setup import utils  # TODO: Move one level up?

# from mlonmcu.timeout import exec_timeout
# from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.target.metrics import Metrics
from mlonmcu.logging import get_logger
from mlonmcu.target.target import Target

# from mlonmcu.models.utils import get_data_source

from ..mlif.mlif import MlifPlatform
from .mlif_litex_target import get_mlif_litex_platform_targets, create_mlif_litex_platform_target

logger = get_logger()


class MlifLitexPlatform(MlifPlatform):
    """Model Library Interface + Litex Platform class."""

    FEATURES = MlifPlatform.FEATURES

    DEFAULTS = {
        **MlifPlatform.DEFAULTS,
    }

    REQUIRED = MlifPlatform.REQUIRED | {"litex.install_dir", "litex.venv", "litex.launcher"}
    OPTIONAL = MlifPlatform.OPTIONAL | {"cmake.exe"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "mlif_litex",
            features=features,
            config=config,
        )
        self.tempdir = None
        self.build_dir = None
        self.litex_name = "sim"

    @property
    def workdir(self):
        return self.build_dir / "build" / self.litex_name

    @property
    def gateware_dir(self):
        return self.workdir / "gateware"

    def create_target(self, name):
        targets = self.get_supported_targets()
        assert name in targets, f"{name} is not a valid MLIF Litex target"
        if name in targets:
            base = targets[name]
        else:
            base = Target
        return create_mlif_litex_platform_target(name, self, base=base)

    def _get_supported_targets(self):
        target_names = get_mlif_litex_platform_targets()
        print("target_names")
        return target_names

    @property
    def mlif_litex_dir(self):
        return Path(self.config["mlif_litex.src_dir"])

    @property
    def litex_install_dir(self):
        return Path(self.config["litex.install_dir"])

    @property
    def litex_launcher(self):
        return Path(self.config["litex.launcher"])

    @property
    def litex_venv_dir(self):
        return Path(self.config["litex.venv"])

    def get_definitions(self):
        definitions = super().get_definitions()
        return definitions

    def get_cmake_args(self, target):
        cmakeArgs = []
        cmakeArgs = super().get_cmake_args(target)
        cmakeArgs.append(f"-DLITEX_ROOT={self.litex_install_dir}")
        cmakeArgs.append(f"-DLITEX_WORKDIR={self.workdir}")
        cmakeArgs.append(f"-DLITEX_CPU={target.litex_cpu}")
        return cmakeArgs

    def _get_soc_gen_args(self, target):
        ret = [
            "--cpu-type",
            target.litex_cpu,
            "--cpu-variant",
            target.litex_cpu_variant,
            "--bus-standard",
            target.bus_standard,
            "--sys-clk-freq",
            str(target.sys_clk_freq),
            "--build",
            "--integrated-main-ram-size",
            hex(target.integrated_main_ram_size),
            "--name",
            self.litex_name,
        ]
        return ret

    def _litex_soc_gen(self, target, **kwargs):
        ret = ""
        soc_gen_args = self._get_soc_gen_args(target)
        env = self.prepare_environment(target)
        ret += utils.execute(
            self.litex_launcher,
            "litex_soc_gen",
            *soc_gen_args,
            cwd=self.build_dir,
            env=env,
            live=self.print_outputs,
            **kwargs,
        )
        return ret

    def _get_sim_args(self, target):
        demo_bin = self.build_dir / "bin" / "generic_mlonmcu.bin"
        ret = [
            "--cpu-type",
            target.litex_cpu,
            "--cpu-variant",
            target.litex_cpu_variant,
            "--no-compile-gateware",
            "--non-interactive",
            "--integrated-main-ram-size",
            hex(target.integrated_main_ram_size),
            "--ram-init",
            demo_bin,
        ]
        return ret

    def _litex_sim(self, target, **kwargs):
        ret = ""
        kwargs.pop("cwd")
        sim_args = self._get_sim_args(target)
        env = self.prepare_environment(target)
        ret += utils.execute(
            self.litex_launcher, "litex_sim", *sim_args, cwd=self.build_dir, env=env, live=self.print_outputs, **kwargs
        )
        return ret

    def _build_vsim(self, target, **kwargs):
        ret = ""
        kwargs.pop("cwd")
        env = self.prepare_environment(target)
        ret += utils.execute(
            self.litex_launcher,
            "bash",
            "build_sim.sh",
            cwd=self.gateware_dir,
            env=env,
            live=self.print_outputs,
            **kwargs,
        )
        return ret

    # def _run_vsim(self, target, **kwargs):
    #     ret = ""
    #     vsim_exe = Path("obj_dir") / "Vsim"
    #     assert vsim_exe.is_file()
    #     env = self.prepare_environment(target)
    #     ret += utils.execute(
    #         self.litex_launcher, vsim_exe, cwd=self.gateware_dir, env=env, live=self.print_outputs, **kwargs
    #     )
    #     return ret

    def prepare(self, target):
        ret = ""
        ret += self._litex_soc_gen(target)
        return ret

    def prepare_environment(self, target=None):
        env = os.environ.copy()
        # TODO: refactor
        if target is not None:
            if target.riscv_gcc_prefix is not None:
                path_old = env["PATH"]
                riscv_prefix = target.riscv_gcc_prefix
                path_new = f"{riscv_prefix}/bin:{path_old}"
                env["PATH"] = path_new
        env["VENV_DIR"] = self.litex_venv_dir
        return env

    def generate(self, src, target, model=None) -> Tuple[dict, dict]:
        _ = self.prepare(target)
        # TODO: store as artifact
        # TODO: measure time?
        artifacts, metrics = super().generate(src, target, model=model)
        if isinstance(artifacts, dict):
            assert len(artifacts) == 1
            artifacts = list(artifacts.values())[0]
        if isinstance(metrics, dict):
            assert len(metrics) == 1
            metrics = list(metrics.values())[0]
        return {"default": artifacts}, {"default": metrics}

    def flash(self, elf, target, timeout=120, **kwargs):
        # TODO: elf to bin?
        self._litex_sim(target, **kwargs)
        self._build_vsim(target, **kwargs)

    def monitor(self, target, timeout=60, **kwargs):
        vsim_exe = self.gateware_dir / "obj_dir" / "Vsim"
        assert vsim_exe.is_file()
        env = self.prepare_environment(target)

        import subprocess
        import signal

        # import time
        # import select
        # import fcntl

        def _kill_monitor():
            pass

        # def _set_nonblock(fd):
        #     flag = fcntl.fcntl(fd, fcntl.F_GETFL)
        #     fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
        #     new_flag = fcntl.fcntl(fd, fcntl.F_GETFL)
        #     assert (new_flag & os.O_NONBLOCK) != 0, "Cannot set file descriptor {fd} to non-blocking"

        # def _await_ready(rlist, wlist, timeout_sec=None, end_time=None):
        #     if timeout_sec is None and end_time is not None:
        #         timeout_sec = max(0, end_time - time.monotonic())

        #     rlist, wlist, xlist = select.select(rlist, wlist, rlist + wlist, timeout_sec)
        #     if not rlist and not wlist and not xlist:
        #         raise RuntimeError("Timeout?")

        #     return True

        def _monitor_helper(verbose=False, start_match=None, end_match=None, timeout=60):
            # start_match and end_match are inclusive
            if timeout:
                pass  # TODO: implement timeout
            outStr = ""
            found_start = start_match is None
            # TODO: log command!
            kwargs.pop("cwd")
            process = subprocess.Popen(
                vsim_exe,
                cwd=self.gateware_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # TODO: store stdout?
                # stdin=subprocess.PIPE,
                bufsize=0,
                **kwargs,
            )
            # _set_nonblock(process.stdin.fileno())

            try:
                exit_code = None
                for line in process.stdout:
                    new_line = line.decode(errors="replace")
                    if verbose:
                        print(new_line.replace("\n", ""))
                    # if not found_menu:
                    #     if menu_match in new_line:
                    #         # print("FOUND MENU")
                    #         found_menu = True
                    #         # process.stdin.write(b"3\n")
                    #         data = b"3"
                    #         fd = process.stdin.fileno()
                    #         _await_ready([], [fd], end_time=None)
                    #         _ = os.write(fd, data)
                    #         # num_written = os.write(fd, data)
                    #         # print("num_written", num_written)
                    # else:
                    if True:
                        if start_match and start_match in new_line:
                            outStr = new_line
                            found_start = True
                        else:
                            outStr = outStr + new_line
                        if found_start:
                            if end_match and end_match in new_line:
                                # _kill_monitor()
                                process.terminate()
                                exit_code = 0
                while exit_code is None:
                    exit_code = process.poll()
                if not verbose and exit_code != 0:
                    logger.error(outStr)
                cmd = "TODO"
                assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
                    exit_code, cmd
                )
            except KeyboardInterrupt:
                logger.debug("Interrupted subprocess. Sending SIGINT signal...")
                _kill_monitor()
                pid = process.pid
                os.kill(pid, signal.SIGINT)
            # os.system("reset")
            return outStr

        logger.debug("Monitoring verilator")
        # TODO: do not drop verilator stdout/stderr?
        return _monitor_helper(
            verbose=self.print_outputs,
            start_match="Program start.",
            end_match="Program finish.",  # TODO: missing exit code?
            timeout=timeout,
        )

    def run(self, elf, target, timeout=120, **kwargs):
        # Only allow one serial communication at a time
        # with FileLock(Path(tempfile.gettempdir()) / "mlonmcu_serial.lock"):
        metrics = Metrics()
        start_time = time.time()
        self.flash(elf, target, timeout=timeout, **kwargs)
        end_time = time.time()
        diff = end_time - start_time
        start_time = time.time()
        output = self.monitor(target, timeout=timeout, **kwargs)
        end_time = time.time()
        diff2 = end_time - start_time
        metrics.add("Verilator Build Time [s]", diff, True)
        metrics.add("Verilator Monitor Time [s]", diff2, True)
        metrics.add("Simulation Time [s]", diff2, True)
        artifacts = []

        return output, metrics, artifacts
