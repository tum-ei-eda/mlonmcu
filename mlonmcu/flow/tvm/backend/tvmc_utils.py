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


def get_runtime_executor_tvmc_args(runtime, executor):
    return ["--runtime", runtime, "--executor", executor]


def get_pass_config_tvmc_args(pass_config):
    args = []
    for key, value in pass_config.items():
        args.extend(["--pass-config", f"{key}={value}"])
    return args


def get_disabled_pass_tvmc_args(disabled_passes):
    args = []
    for item in disabled_passes:
        args.extend(["--disabled-pass", item])
    return args


def get_input_shapes_tvmc_args(input_shapes):
    if input_shapes is None:
        return []
    arg = " ".join([f"{name}:[" + ",".join(list(map(str, dims))) + "]" for name, dims in input_shapes.items()])
    return ["--input-shapes", arg]


def gen_target_details_args(target, target_details):
    return sum([[f"--target-{target}-{key}", value] for key, value in target_details.items()], [])


def get_target_tvmc_args(target="c", extra_target=None, target_details={}, extra_target_details=None):
    if extra_target:
        if isinstance(extra_target, str):
            if "," in extra_target:
                extra_target = extra_target.split(",")
            else:
                extra_target = [extra_target]
        # TODO: support multiple ones, currently only single one...
        assert len(extra_target) == 1
        target = ",".join(extra_target + [target])
    return [
        "--target",
        target,
        # TODO: provide a feature which sets these automatically depending on the chosen target
        *gen_target_details_args(target, target_details),
        *(gen_target_details_args(extra_target[0], extra_target_details) if extra_target is not None else []),
    ]


def get_tuning_records_tvmc_args(use_tuning_results, tuning_records_file):
    if use_tuning_results:
        assert tuning_records_file is not None, "No tuning records are available"
        return ["--tuning-records", str(tuning_records_file)]
    else:
        return []


def get_rpc_tvmc_args(enabled, key, hostname, port):
    return (
        [
            "--rpc-key",
            key,
            "--rpc-tracker",
            hostname + ":" + str(port),
        ]
        if enabled
        else []
    )


def get_tvmaot_tvmc_args(alignment_bytes, unpacked_api):
    return [
        *["--runtime-crt-system-lib", str(0)],
        *["--target-c-constants-byte-alignment", str(alignment_bytes)],
        *["--target-c-workspace-byte-alignment", str(alignment_bytes)],
        *["--target-c-executor", "aot"],
        *["--target-c-unpacked-api", str(int(unpacked_api))],
        *["--target-c-interface-api", "c" if unpacked_api else "packed"],
    ]


def get_tvmrt_tvmc_args():
    return [
        *["--runtime-crt-system-lib", str(1)],
        *["--executor-graph-link-params", str(0)],
    ]


def get_data_tvmc_args(mode=None, ins_file=None, outs_file=None, print_top=10):
    ret = []
    if ins_file is not None:
        ret.extend(["--inputs", ins_file])
    else:
        assert mode is not None
        ret.extend(["--fill-mode", mode])

    if outs_file is not None:
        ret.extend(["--outputs", outs_file])

    if print_top is not None and print_top > 0:
        ret.extend(["--print-top", print_top])

    return ret


def get_bench_tvmc_args(print_time=False, profile=False, end_to_end=False, repeat=1, number=1):
    ret = []

    if print_time:
        ret.append("--print-time")

    if profile:
        ret.append("--profile")

    if end_to_end:
        ret.append("--end-to-end")

    if repeat:
        ret.extend(["--repeat", str(repeat)])

    if number:
        ret.extend(["--number", str(number)])

    return ret
