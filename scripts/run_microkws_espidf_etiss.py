################# How to run ################
# Inside mlonmcu parent directory
# Ensure MLonMCU_HOME path is set in shell
# export MLONMCU_HOME=/nas/ei/share/TUEIEDAprojects/SystemDesign/work/performance-evaluation/varghese/mlonmcu_env
# Enter virtual environment
# source .venv/bin/activate
# python3 scripts/run_microkws_espidf_etiss.py -m <mode>
#############################################


import argparse
import subprocess
import textwrap
import csv
import os

# These can be overwritten by defining environement variables
MICRO_KWS_MODEL = os.environ.get(
    "MICRO_KWS_MODEL",
    "/usr/local/research/projects/SystemDesign/work/performance-evaluation/varghese/Micro-KWS/micro-kws/2_deploy/data/micro_kws_student_quantized.tflite",
)
ESPIDF_INSTALL = os.environ.get(
    "ESPIDF_INSTALL", "/usr/local/research/projects/SystemDesign/tools/esp/v5.4.1/espressif_py310"
)
ESPIDF_SRC = os.environ.get("ESPIDF_SRC", "/usr/local/research/projects/SystemDesign/tools/esp/v5.4.1/esp-idf")
AUTOTUNED_RESULTS = os.environ.get(
    "AUTOTUNED_RESULTS",
    "/usr/local/research/projects/SystemDesign/work/performance-evaluation/varghese/Micro-KWS/micro-kws/2_deploy/data/micro_kws_student_tuning_log_nchw_best.txt",
)
ESP32C3_GCC_INSTALL = os.environ.get(
    "ESP32C3_GCC_INSTALL",
    "/usr/local/research/projects/SystemDesign/tools/esp/v5.4.1/espressif_py310/tools/riscv32-esp-elf/esp-14.2.0_20241119/riscv32-esp-elf",
)
ETISS_SRC = os.environ.get(
    "ETISS_SRC",
    "/usr/local/research/projects/SystemDesign/work/performance-evaluation/varghese/PerformanceSimulation_workspace/etiss-perf-sim/etiss/",
)
ETISS_INSTALL = os.environ.get(
    "ETISS_INSTALL",
    "/usr/local/research/projects/SystemDesign/work/performance-evaluation/varghese/PerformanceSimulation_workspace/etiss-perf-sim/etiss/build_dir/installed",
)
ETISS_EXE = os.environ.get(
    "ETISS_EXE",
    "/usr/local/research/projects/SystemDesign/work/performance-evaluation/varghese/PerformanceSimulation_workspace/etiss-perf-sim/etiss/build_dir/installed/bin/bare_etiss_processor",
)


def get_results_csv_path():
    mlonmcu_home = os.environ.get("MLONMCU_HOME")
    if not mlonmcu_home:
        raise EnvironmentError("MLONMCU_HOME environment variable is not set.")
    return os.path.join(mlonmcu_home, "temp", "sessions", "latest", "report.csv")


def parse_simulation_results(rows):
    row = rows[-1]  # Last row
    run_cycles = int(row["Run Cycles"])
    run_instructions = int(row["Run Instructions"])
    run_cpi = float(row["Run CPI"])
    return run_cycles, run_instructions, run_cpi


def parse_esp32_perf_results(rows):
    # Note that the rows could get interchanged if the PCER_INIT_VAL
    # is given in the opposite order than what it is currently
    # Check cmd in esp32_perf mode
    # The correct order is {'PCER_INIT_VAL': 1}, then {'PCER_INIT_VAL': 2}
    run_cycles = int(rows[0]["Run Cycles"])
    run_instructions = int(rows[1]["Run Instructions"])
    run_cpi = run_cycles / run_instructions if run_instructions else float("inf")
    return run_cycles, run_instructions, run_cpi


def parse_results_csv(mode):
    results_path = get_results_csv_path()
    with open(results_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    if not rows:
        raise ValueError("No data found in results.csv")

    if mode == "etiss_sim":
        print("Results from Simulation")
        return parse_simulation_results(rows)
    elif mode == "esp32_perf":
        print("Results from Hardware")
        return parse_esp32_perf_results(rows)
    elif mode == "esp32_real":
        return None
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def run_command(command, mode, opt):
    print("Running command...\n")
    print(command, "\n")
    result = subprocess.run(command, shell=True)
    if result.returncode == 0:
        print("\nCommand executed successfully.\n")
        result = parse_results_csv(mode)
        if result:
            run_cycles, run_instructions, run_cpi = result
            print(f"Compiler Optimization: {opt}")
            print(f"Model Run Cycles: {run_cycles}")
            print(f"Model Run Instructions: {run_instructions}")
            print(f"Model Run CPI: {run_cpi:.6f}")
        return result
    else:
        print(f"\nCommand failed with return code: {result.returncode}")
        print("Error Output:\n", result.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Run MLonMCU flows for ESP32 or Simulator")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["esp32_perf", "etiss_sim", "esp32_real", "compare_esp32_and_etiss"],
        required=True,
        help="Choose the run mode: esp32_perf, sim, esp32_real, or compare_esp32_and_etiss",
    )
    parser.add_argument(
        "-w", "--wait", type=int, choices=[0, 1], default=1, help="Set espidf.wait_for_user to 1 or 0 (default: 0)"
    )
    parser.add_argument(
        "-o", "--opt", choices=["0", "2", "s"], default="s", help="Compiler optimization level: 0, 2, or s (default: s)"
    )
    args = parser.parse_args()

    cmd_esp32_perf = textwrap.dedent(
        f"""\
        python3 -m mlonmcu.cli.main flow run {MICRO_KWS_MODEL} \
        --target esp32c3 --platform espidf \
        -c espidf.install_dir={ESPIDF_INSTALL} -c espidf.src_dir={ESPIDF_SRC} \
        -c espidf.print_outputs=1 -c esp32c3.print_outputs=1 -c run.export_optional=1 \
        --backend tvmaotplus -c tvmaotplus.desired_layout=NCHW -c tvmaot.desired_layout=NCHW \
        -f autotuned -c autotuned.results_file={AUTOTUNED_RESULTS} \
        -c espidf.project_template=micro_kws_esp32devboard_perf -c espidf.wait_for_user={args.wait} \
        -c riscv_gcc_rv32.install_dir={ESP32C3_GCC_INSTALL} -c riscv_gcc_rv32.name=riscv32-esp-elf \
        -c espidf.extra_cmake_defs="{{'CMAKE_C_FLAGS': '-march=rv32im_zicsr_zifencei', 'CMAKE_ASM_FLAGS': '-march=rv32im_zicsr_zifencei', 'CMAKE_CXX_FLAGS': '-march=rv32im_zicsr_zifencei', 'CMAKE_EXE_LINKER_FLAGS': '-nostartfiles -march=rv32im_zicsr_zifencei --specs=nosys.specs'}}" \
        --config-gen espidf.extra_cmake_defs="{{'PCER_INIT_VAL': 1, 'ENABLE_PERF_EVAL': 1}}" \
        --config-gen espidf.extra_cmake_defs="{{'PCER_INIT_VAL': 2, 'ENABLE_PERF_EVAL': 1}}" \
        -c espidf.optimize={args.opt}
    """
    )

    cmd_etiss_sim = textwrap.dedent(
        f"""\
        python3 -m mlonmcu.cli.main flow run {MICRO_KWS_MODEL} \
        --target etiss_perf -c run.export_optional=1 \
        -c espidf.install_dir={ESPIDF_INSTALL} -c espidf.src_dir={ESPIDF_SRC} \
        --backend tvmaotplus -c tvmaotplus.desired_layout=NCHW -c tvmaot.desired_layout=NCHW \
        -f autotuned -c autotuned.results_file={AUTOTUNED_RESULTS} \
        -c riscv_gcc_rv32.install_dir={ESP32C3_GCC_INSTALL} -c riscv_gcc_rv32.name=riscv32-esp-elf \
        -c etiss_perf.src_dir={ETISS_SRC} -c etiss_perf.install_dir={ETISS_INSTALL} -c etiss_perf.exe={ETISS_EXE} \
        -c etiss_perf.fpu=none -c etiss_perf.atomic=0 -c etiss_perf.compressed=0 \
        -f perf_sim -c mlif.optimize={args.opt} -c perf_sim.core=esp32c3 -c etiss_perf.flash_start=0x42000000 -c etiss_perf.flash_size=0x800000
    """
    )

    cmd_esp32_real = textwrap.dedent(
        f"""\
        python3 -m mlonmcu.cli.main flow run {MICRO_KWS_MODEL} \
        --target esp32c3 --platform espidf \
        -c espidf.install_dir={ESPIDF_INSTALL} -c espidf.src_dir={ESPIDF_SRC} \
        -c espidf.print_outputs=1 -c esp32c3.print_outputs=1 -c run.export_optional=1 \
        --backend tvmaotplus -c tvmaotplus.desired_layout=NCHW -c tvmaot.desired_layout=NCHW \
        -f autotuned -c autotuned.results_file={AUTOTUNED_RESULTS} \
        -c espidf.project_template=micro_kws_esp32devboard_perf -c espidf.wait_for_user={args.wait} \
        -c riscv_gcc_rv32.install_dir={ESP32C3_GCC_INSTALL} -c riscv_gcc_rv32.name=riscv32-esp-elf \
        -c espidf.optimize={args.opt} -c espidf.extra_cmake_defs="{{'CONFIG_ENABLE_WIFI': 1}}"
    """
    )

    if args.mode == "esp32_perf":
        run_command(cmd_esp32_perf, args.mode, args.opt)
    elif args.mode == "etiss_sim":
        run_command(cmd_etiss_sim, args.mode, args.opt)
    elif args.mode == "esp32_real":
        run_command(cmd_esp32_real, args.mode, args.opt)
    elif args.mode == "compare_esp32_and_etiss":
        result_esp32 = run_command(cmd_esp32_perf, "esp32_perf", args.opt)
        result_etiss = run_command(cmd_etiss_sim, "etiss_sim", args.opt)
        if result_esp32 and result_etiss:
            # Display both results in a comparable manner
            esp32_cycles, esp32_instructions, esp32_cpi = result_esp32
            etiss_cycles, etiss_instructions, etiss_cpi = result_etiss

            print("\n\n===== Comparative Results =====\n")
            print(f"{'Metric':<25}{'ESP32-C3':>15}{'ETISS Simulator':>20}{'Difference (%)':>20}")
            print("-" * 80)

            def percent_diff(val1, val2):
                return ((val2 - val1) / val1) * 100 if val1 else float("inf")

            def norm(val1, val2):
                return (val1 / val2) * 100 if val1 else float("inf")

            print(
                f"{'Run Cycles':<25}{esp32_cycles:>15}{etiss_cycles:>20}{percent_diff(esp32_cycles, etiss_cycles):>19.2f}%"
            )
            print(
                f"{'Run Instructions':<25}{esp32_instructions:>15}{etiss_instructions:>20}{percent_diff(esp32_instructions, etiss_instructions):>19.2f}%"
            )
            print(f"{'Run CPI':<25}{esp32_cpi:>15.6f}{etiss_cpi:>20.6f}{percent_diff(esp32_cpi, etiss_cpi):>19.2f}%")
            print("-" * 80)
            print("\n\n===== Benefit w.r.t CPI=1 =====\n")
            iss_cpi = 1
            iss_cpi_err = percent_diff(esp32_cpi, iss_cpi)
            iss_cpi_err_norm = norm(iss_cpi_err, iss_cpi_err)
            etiss_cpi_err = percent_diff(esp32_cpi, etiss_cpi)
            etiss_cpi_err_norm = norm(etiss_cpi_err, iss_cpi_err)
            print(
                f"{'Device':<25}{'Instructions':>15}{'Cycles':>20}{'CPI':>20}{'CPI Error (%)':>20}{'Norm. CPI Error (%)':>25}{'Benefit (%)':>20}"
            )
            print("-" * 145)
            print(
                f"{'ESP32-C3':<25}{esp32_instructions:>15}{esp32_cycles:>20}{esp32_cpi:>20.6f}{percent_diff(esp32_cpi, esp32_cpi):>19.2f}%{percent_diff(esp32_cpi, esp32_cpi):>24.2f}%"
            )
            print(
                f"{'ISS (CPI = 1)':<25}{etiss_instructions:>15}{etiss_instructions:>20}{iss_cpi:>20.6f}{iss_cpi_err:>19.2f}%{iss_cpi_err_norm:>24.2f}%{iss_cpi_err_norm-iss_cpi_err_norm:>19.2f}%"
            )
            print(
                f"{'ETISS':<25}{etiss_instructions:>15}{etiss_cycles:>20}{etiss_cpi:>20.6f}{etiss_cpi_err:>19.2f}%{etiss_cpi_err_norm:>24.2f}%{iss_cpi_err_norm-etiss_cpi_err_norm:>19.2f}%"
            )
            print("-" * 145)


if __name__ == "__main__":
    main()
