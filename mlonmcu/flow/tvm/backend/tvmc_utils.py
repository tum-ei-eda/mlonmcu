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
    if len(disabled_passes) == 0:
        return []
    arg = ",".join(disabled_passes)
    return ["--disabled-pass", arg]


def get_input_shapes_tvmc_args(input_shapes):
    if input_shapes is None:
        return []
    arg = " ".join([f"{name}:[" + ",".join(list(map(str, dims))) + "]" for name, dims in input_shapes.items()])
    return ["--input-shapes", arg]


def check_allowed(target, name):
    common = ["libs", "model", "tag", "mcpu", "device", "keys"]
    if target == "c":
        return name in ["constants-byte-alignment", "workspace-bytes-alignment", "march"] + common
    elif target == "llvm":
        return (
            name
            in [
                "fast-math",
                "opt-level",
                "fast-math-ninf",
                "mattr",
                "num-cores",
                "fast-math-nsz",
                "fast-math-contract",
                "mtriple",
                "mfloat-abi",
                "fast-math-arcp",
                "fast-math-reassoc",
                "mabi",
                "num_cores",
            ]
            + common
        )

    else:
        return True


def gen_target_details_args(target, target_details):
    def helper(value):
        if isinstance(value, (bool, int)):
            # value = "true" if value else "false"
            value = str(int(value))
        return value

    return sum(
        [
            [f"--target-{target}-{key}", helper(value)]
            for key, value in target_details.items()
            if check_allowed(target, key)
        ],
        [],
    )


def gen_extra_target_details_args(extra_target_details):
    ret = []
    for extra_target, target_details in extra_target_details.items():
        if target_details:
            ret.append(gen_target_details_args(extra_target, target_details))
    return sum(ret, [])


def get_target_tvmc_args(target="c", extra_targets=[], target_details={}, extra_target_details={}):
    if extra_targets:
        assert isinstance(extra_targets, list)
    else:
        extra_targets = []
    if extra_target_details:
        assert isinstance(extra_target_details, dict)
    else:
        extra_target_details = {}

    return [
        "--target",
        ",".join(extra_targets + [target]),
        # TODO: provide a feature which sets these automatically depending on the chosen target
        *gen_target_details_args(target, target_details),
        *(gen_extra_target_details_args(extra_target_details)),
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


def get_tvmaot_tvmc_args(alignment_bytes, unpacked_api, runtime="crt", target="c", system_lib=False):
    ret = []
    if runtime == "crt":
        if unpacked_api:
            assert not system_lib, "Unpacked API is incompatible with system lib"
        ret += ["--runtime-crt-system-lib", str(int(system_lib))]

    ret += [
        *["--executor-aot-unpacked-api", str(int(unpacked_api))],
        *["--executor-aot-interface-api", "c" if unpacked_api else "packed"],
    ]
    if target == "c":
        ret += [
            *["--target-c-constants-byte-alignment", str(alignment_bytes)],
            *["--target-c-workspace-byte-alignment", str(alignment_bytes)],
        ]
    return ret


def get_tvmrt_tvmc_args(runtime="crt", system_lib=True, link_params=True):
    ret = []
    if runtime == "crt":
        ret += ["--runtime-crt-system-lib", str(int(system_lib))]
    ret += ["--executor-graph-link-params", str(int(link_params))]
    return ret


def get_data_tvmc_args(mode=None, ins_file=None, outs_file=None, print_top=None):
    ret = []
    if ins_file is not None:
        ret.extend(["--inputs", ins_file])
    else:
        if mode is not None:
            ret.extend(["--fill-mode", mode])

    if outs_file is not None:
        ret.extend(["--outputs", outs_file])

    if print_top is not None and print_top > 0:
        ret.extend(["--print-top", str(print_top)])

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


def get_desired_layout_args(layouts, ops, mapping):
    if mapping:
        assert layouts is None, "desired_layout not allowed when using desired_layouts_map"
        assert ops is None, "desired_layout_ops not allowed when using desired_layouts_map"
        layouts = mapping.values()
        ops = mapping.keys()

    if layouts is None:
        layouts = []

    if ops is None:
        ops = []

    if layouts and ops:
        assert len(layouts) == len(ops) or len(layouts) == 1

    ret = []
    if layouts:
        ret.extend(["--desired-layout", *layouts])

    if ops:
        ret.extend(["--desired-layout-ops", *ops])

    return ret
