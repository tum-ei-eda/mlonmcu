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
"""Collection of (example) postprocesses integrated in MLonMCU."""

import re
import ast
import tempfile
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd

from mlonmcu.artifact import Artifact, ArtifactFormat, lookup_artifacts
from mlonmcu.config import str2dict, str2bool, str2list
from mlonmcu.logging import get_logger

from .postprocess import SessionPostprocess, RunPostprocess
from .validate_metrics import parse_validate_metrics, parse_classify_metrics

logger = get_logger()


def match_rows(df, cols):
    """Helper function to group similar rows in a dataframe."""
    groups = df.astype(str).groupby(cols).apply(lambda x: tuple(x.index)).tolist()
    return groups


def _check_cfg(value):
    res = re.compile(r"^((?:[a-zA-Z\d\-_ \.\[\]]+)(?:,[a-zA-Z\d\-_ \.\[\]]+)*)$").match(value)
    if res is None:
        return False
    return True


def _parse_cfg(value):
    if _check_cfg(value):
        return value.split(",")
    else:
        return ast.literal_eval(value)


class FilterColumnsPostprocess(SessionPostprocess):
    """Postprocess which can be used to drop unwanted columns from a report."""

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "keep": None,
        "drop": None,
        "drop_nan": False,
        "drop_empty": False,
        "drop_const": False,
    }

    def __init__(self, features=None, config=None):
        super().__init__("filter_cols", features=features, config=config)

    @property
    def keep(self):
        """Get keep property."""
        cfg = self.config["keep"]
        if isinstance(cfg, str):
            return _parse_cfg(cfg)
        return cfg

    @property
    def drop(self):
        """Get drop property."""
        cfg = self.config["drop"]
        if isinstance(cfg, str):
            return _parse_cfg(cfg)
        return cfg

    @property
    def drop_nan(self):
        """Get drop_nan property."""
        value = self.config["drop_nan"]
        return str2bool(value)

    @property
    def drop_empty(self):
        """Get drop_empty property."""
        value = self.config["drop_empty"]
        return str2bool(value)

    @property
    def drop_const(self):
        """Get drop_const property."""
        value = self.config["drop_const"]
        return str2bool(value)

    def post_session(self, report):
        """Called at the end of a session."""

        def _filter_df(df, keep, drop, drop_nan=False, drop_empty=False, drop_const=False):
            if drop_empty:
                raise NotImplementedError
            if drop_nan:
                df.dropna(axis=1, how="all", inplace=True)
            if drop_const:
                df = df.loc[:, (df != df.iloc[0]).any()]
            if not (keep is None or drop is None):
                raise RuntimeError("'drop' and 'keep' can not be defined at the same time")
            if keep is not None:
                drop_cols = [name for name in df.columns if name not in keep]
            elif drop is not None:
                drop_cols = [name for name in df.columns if name in drop]
            else:
                drop_cols = []
            return df.drop(columns=drop_cols)

        report.pre_df = _filter_df(
            report.pre_df,
            self.keep,
            self.drop,
            drop_nan=self.drop_nan,
            drop_empty=self.drop_empty,
            drop_const=self.drop_const,
        )
        report.main_df = _filter_df(
            report.main_df, self.keep, self.drop, drop_nan=self.drop_nan, drop_const=self.drop_const
        )
        report.post_df = _filter_df(
            report.post_df, self.keep, self.drop, drop_nan=self.drop_nan, drop_const=self.drop_const
        )


class RenameColumnsPostprocess(SessionPostprocess):
    """Postprocess which can rename columns based on a provided mapping."""

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "mapping": {},
        "merge": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("rename_cols", features=features, config=config)

    @property
    def mapping(self):
        value = self.config["mapping"]
        if not isinstance(value, dict):
            return str2dict(value)
        return value

    @property
    def merge(self):
        value = self.config["merge"]
        return str2bool(value)

    def post_session(self, report):
        """Called at the end of a session."""
        values = self.mapping.values()
        if len(values) != len(set(values)) and not self.merge:
            logger.warning("rename_cols: non unique mapping found. use merge=True to avoid overwriting values.")

        def merge(df):
            if len(set(df.columns)) == len(df.columns):
                return df
            a = df.loc[:, ~df.columns.duplicated(keep="first")]
            b = df.loc[:, df.columns.duplicated(keep="first")]
            return a.combine_first(merge(b))

        report.pre_df = report.pre_df.rename(columns=self.mapping)
        report.main_df = report.main_df.rename(columns=self.mapping)
        report.post_df = report.post_df.rename(columns=self.mapping)

        if self.merge:
            report.pre_df = merge(report.pre_df)
            report.main_df = merge(report.main_df)
            report.post_df = merge(report.post_df)


class Features2ColumnsPostprocess(SessionPostprocess):  # RunPostprocess?
    """Postprocess which can be used to transform (explode) the 'Features' Column
    in a dataframe for easier filtering."""

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "limit": [],
        "drop": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("features2cols", features=features, config=config)

    @property
    def limit(self):
        """Get limit property."""
        value = self.config["limit"]
        if not isinstance(value, list):
            return str2list(value)
        return value

    @property
    def drop(self):
        """Get drop property."""
        value = self.config["drop"]
        return str2bool(value)

    def post_session(self, report):
        df = report.post_df
        if "Features" not in df.columns:
            return
        to_concat = [
            df["Features"].apply(lambda x: pd.Series({"feature_" + feature_name: feature_name in x}))
            for feature_name in list(set(df["Features"].sum()))
            if feature_name in self.limit or len(self.limit) == 0
        ]
        if len(to_concat) == 0:
            return
        feature_df = pd.concat(
            to_concat,
            axis=1,
        )
        if self.drop:
            tmp_df = df.drop(columns=["Features"])
        else:
            tmp_df = df
        new_df = pd.concat([tmp_df, feature_df], axis=1)
        report.post_df = new_df


class Config2ColumnsPostprocess(SessionPostprocess):  # RunPostprocess?
    """Postprocess which can be used to transform (explode) the 'Config' Column in a dataframe for easier filtering."""

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "limit": [],
        "drop": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("config2cols", features=features, config=config)

    @property
    def limit(self):
        """Get limit property."""
        value = self.config["limit"]
        if not isinstance(value, list):
            return str2list(value)
        return value

    @property
    def drop(self):
        """Get drop property."""
        value = self.config["drop"]
        return str2bool(value)

    def post_session(self, report):
        """Called at the end of a session."""
        df = report.post_df
        if "Config" not in df.columns:
            return
        config_df = (
            df["Config"]
            .apply(lambda x: {key: value for key, value in x.items() if key in self.limit or len(self.limit) == 0})
            .apply(pd.Series)
            .add_prefix("config_")
        )
        if self.drop:
            tmp_df = df.drop(columns=["Config"])
        else:
            tmp_df = df
        new_df = pd.concat([tmp_df, config_df], axis=1)
        report.post_df = new_df


class MyPostprocess(SessionPostprocess):
    """TODO"""

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
    }

    def __init__(self, features=None, config=None):
        super().__init__("mypost", features=features, config=config)
        self.config2cols = Config2ColumnsPostprocess(
            config={
                "config2cols.limit": [
                    "tvmllvm.desired_layout",
                    "tvmaot.desired_layout",
                    "tvmaotplus.desired_layout",
                    "tvmrt.desired_layout",
                    "xcorev.mem",
                    "xcorev.mac",
                    "xcorev.bi",
                    "xcorev.alu",
                    "xcorev.bitmanip",
                    "xcorev.simd",
                    "xcorev.hwlp",
                    "cv32e40p.fpu",
                    "etiss.fpu",
                    "corev_ovpsim.fpu",
                    "tvmaot.disabled_passes",
                    "tvmaotplus.disabled_passes",
                    "tvmrt.disabled_passes",
                    "tvmllvm.disabled_passes",
                    "auto_vectorize.loop",
                    "auto_vectorize.slp",
                    "auto_vectorize.force_vector_width",
                    "auto_vectorize.force_vector_interleave",
                    "auto_vectorize.custom_unroll",
                    "tvmllvm.target_keys",
                    "tvmrt.target_keys",
                    "tvmaot.target_keys",
                    "tvmaotplus.target_keys",
                    "autotuned.mode",
                ],
                "config2cols.drop": True,
            }
        )
        self.rename_cols = RenameColumnsPostprocess(
            config={
                "rename_cols.mapping": {
                    "config_tvmllvm.desired_layout": "Layout",
                    "config_tvmaot.desired_layout": "Layout",
                    "config_tvmaotplus.desired_layout": "Layout",
                    "config_tvmrt.desired_layout": "Layout",
                    "config_xcorev.mem": "XCVMem",
                    "config_xcorev.mac": "XCVMac",
                    "config_xcorev.bi": "XCVBi",
                    "config_xcorev.alu": "XCVAlu",
                    "config_xcorev.bitmanip": "XCVBitmanip",
                    "config_xcorev.simd": "XCVSimd",
                    "config_xcorev.hwlp": "XCVHwlp",
                    "feature_autotuned": "Autotuned",
                    "feature_debug": "Debug",
                    "config_cv32e40p.fpu": "FPU",
                    "config_etiss.fpu": "FPU",
                    "config_corev_ovpsim.fpu": "FPU",
                    "config_tvmaot.disabled_passes": "Disabled",
                    "config_tvmaotplus.disabled_passes": "Disabled",
                    "config_tvmrt.disabled_passes": "Disabled",
                    "config_tvmllvm.disabled_passes": "Disabled",
                    "config_auto_vectorize.loop": "Loop",
                    "config_auto_vectorize.slp": "Slp",
                    "config_auto_vectorize.force_vector_width": "FVW",
                    "config_auto_vectorize.force_vector_interleave": "FVI",
                    "config_auto_vectorize.custom_unroll": "Unroll",
                    "config_tvmllvm.target_keys": "Keys",
                    "config_tvmrt.target_keys": "Keys",
                    "config_tvmaot.target_keys": "Keys",
                    "config_tvmaotplus.target_keys": "Keys",
                    "config_autotuned.mode": "Tuner",
                }
            }
        )
        self.features2cols = Features2ColumnsPostprocess(
            config={
                "features2cols.limit": ["autotuned", "debug", "auto_vectorize", "target_optimized"],
                "features2cols.drop": True,
            }
        )
        self.filter_cols = FilterColumnsPostprocess(
            config={
                "filter_cols.drop": [
                    "Postprocesses",
                    "Framework",
                    "Platform",
                    "Session",
                    "ROM read-only",
                    "ROM code",
                    "ROM misc",
                    "RAM data",
                    "RAM zero-init data",
                    "Run Stage Time [s]",
                    "Compile Stage Time [s]",
                    "Workspace Size [B]",
                    "Build Stage Time [s]",
                    "Load Stage Time [s]",
                    "feature_auto_vectorize",
                    "feature_target_optimized",
                    "Setup Cycles",
                    "Setup Instructions",
                    "Setup CPI",
                ]
            }
        )

    def post_session(self, report):
        """TODO"""
        self.config2cols.post_session(report)
        self.features2cols.post_session(report)
        self.rename_cols.post_session(report)
        self.filter_cols.post_session(report)


class PassConfig2ColumnsPostprocess(SessionPostprocess):
    """Postprocess which can be used to transform (explode) the TVM pass_config into separate columns.
    requires prior Config2Columns pass."""

    def __init__(self, features=None, config=None):
        super().__init__("passcfg2cols", features=features, config=config)

    def post_session(self, report):
        """Called at the end of a session."""
        df = report.post_df
        name = "config_tvmaot.extra_pass_config"
        if name not in df.columns:
            return
        config_df = df[name].apply(pd.Series).add_prefix("passcfg_")
        tmp_df = df.drop(columns=[name])
        new_df = pd.concat([tmp_df, config_df], axis=1)
        report.post_df = new_df


class Bytes2kBPostprocess(SessionPostprocess):  # RunPostprocess?
    """Postprocess which can be used to scale the memory related columns from Bytes to KiloBytes."""

    def __init__(self, features=None, config=None):
        super().__init__("bytes2kb", features=features, config=config)

    def post_session(self, report):
        """Called at the end of a session."""
        df = report.main_df
        match_strs = ["ROM", "RAM"]
        cols = list(
            filter(lambda x: any(s in x for s in match_strs), df.columns)
        )  # Only scale columns related to memory
        cols = [col for col in cols if "kB" not in col]  # Do not scale columns with are already in kB

        for col in cols:
            df[col] = df[col] / 1000.0
            df.rename(columns={col: col + " [kB]"}, inplace=True)

        report.main_df = df


class VisualizePostprocess(SessionPostprocess):
    """A very simple example on how to generate a plot of the results using a postprocess."""

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "format": "png",
    }

    def __init__(self, features=None, config=None):
        super().__init__("visualize", features=features, config=config)

    @property
    def format(self):
        """Get format property."""
        return self.config["format"]

    def post_session(self, report):
        """Called at the end of a session."""
        df = pd.concat([report.pre_df, report.main_df], axis=1)

        if self.format != "png":
            raise NotImplementedError("Currently only supports PNG")

        COLS = ["Cycles", "Total ROM", "Total RAM"]
        for col in COLS:
            if col not in report.main_df.columns:
                return []

        # Local import to deal with optional dependencies
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(ncols=len(COLS))
        plt.rcParams["figure.figsize"] = (15, 3)  # (w, h)
        for i, col in enumerate(COLS):
            new_df = df[[col]].astype(float)
            bar_names_df = (
                df["Session"].astype(str) + "_" + df["Run"].astype(str)
            )  # ideally we would use model/backend/target names here...
            new_df.index = bar_names_df
            new_df.plot(kind="bar", ax=axes[i])

        data = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            fig_path = Path(tmpdirname) / "plot.png"
            fig.savefig(fig_path)
            with open(fig_path, "rb") as handle:
                data = handle.read()

        artifacts = [Artifact("plot.png", raw=data, fmt=ArtifactFormat.RAW)]
        return artifacts


class Artifact2ColumnPostprocess(RunPostprocess):
    """Postprocess for converting artifacts to columns in the report."""

    DEFAULTS = {
        **RunPostprocess.DEFAULTS,
        "file2colname": {},
    }

    def __init__(self, features=None, config=None):
        super().__init__("artifacts2cols", features=features, config=config)

    @property
    def file2colname(self):
        """Get file2colname property."""
        value = self.config["file2colname"]
        if not isinstance(value, dict):
            return str2dict(value)
        return value

    def post_run(self, report, artifacts):
        """Called at the end of a run."""
        for filename, colname in self.file2colname.items():
            filename = Path(filename)
            filecol = None
            if ":" in filename.name:
                fname, filecol = filename.name.rsplit(":", 1)
                filename = filename.parent / fname
            matches = lookup_artifacts(artifacts, name=filename, first_only=True)
            if not matches:
                report.main_df[colname] = ""
                continue
            if matches[0].fmt != ArtifactFormat.TEXT:
                raise RuntimeError("Can only put text into report columns")
            content = matches[0].content
            if filecol:
                assert filename.suffix == ".csv"
                filedf = pd.read_csv(StringIO(content))
                if filecol == "*":
                    cols = list(filedf.columns)
                else:
                    assert filecol in filedf.columns
                    cols = [filecol]
                content = filedf[cols].to_dict(orient="list")
                if len(content) == 1:
                    content = content[list(content.keys())[0]]
                if len(content) == 1:
                    content = content[0]
                content = str(content)
            report.main_df[colname] = content
        return []


class AnalyseInstructionsPostprocess(RunPostprocess):
    """Counting specific types of instructions."""

    DEFAULTS = {
        **RunPostprocess.DEFAULTS,
        "groups": True,
        "sequences": True,
        "seq_depth": 3,
        "top": 10,
        "to_df": False,
        "to_file": True,
        "corev": False,
    }

    def __init__(self, features=None, config=None):
        super().__init__("analyse_instructions", features=features, config=config)

    @property
    def groups(self):
        """Get groups property."""
        value = self.config["groups"]
        return str2bool(value)

    @property
    def sequences(self):
        """get sequences property."""
        value = self.config["sequences"]
        return str2bool(value)

    @property
    def seq_depth(self):
        """get seq_depth property."""
        return int(self.config["seq_depth"])

    @property
    def top(self):
        """get top property."""
        return int(self.config["top"])

    @property
    def to_df(self):
        """Get to_df property."""
        value = self.config["to_df"]
        return str2bool(value)

    @property
    def to_file(self):
        """Get to_file property."""
        value = self.config["to_file"]
        return str2bool(value)

    @property
    def corev(self):
        """Get corev property."""
        value = self.config["corev"]
        return str2bool(value)

    def post_run(self, report, artifacts):
        """Called at the end of a run."""
        ret_artifacts = []
        log_artifact = lookup_artifacts(artifacts, flags=("log_instrs",), fmt=ArtifactFormat.TEXT, first_only=True)
        assert len(log_artifact) == 1, "To use analyse_instructions process, please enable feature log_instrs."
        log_artifact = log_artifact[0]
        is_spike = "spike" in log_artifact.flags
        is_etiss = "etiss_pulpino" in log_artifact.flags or "etiss" in log_artifact.flags
        is_ovpsim = "ovpsim" in log_artifact.flags or "corev_ovpsim" in log_artifact.flags
        is_riscv = is_spike or is_etiss or is_ovpsim
        if is_spike:
            content = log_artifact.content
            if self.groups:
                encodings = re.compile(r"\((0x[0-9abcdef]+)\)").findall(content)
            if self.sequences:
                names = re.compile(r"core\s+\d+:\s0x[0-9abcdef]+\s\(0x[0-9abcdef]+\)\s([\w.]+).*").findall(content)
        elif is_etiss:

            # TODO: generalize
            def transform_df(df):
                df["pc"] = df["pc"].apply(lambda x: int(x, 0))
                df["pc"] = pd.to_numeric(df["pc"])
                # TODO: normalize instr names
                df[["instr", "rest"]] = df["rest"].str.split(" # ", n=1, expand=True)
                df["instr"] = df["instr"].apply(lambda x: x.strip())
                df["instr"] = df["instr"].astype("category")
                df[["bytecode", "operands"]] = df["rest"].str.split(" ", n=1, expand=True)

                def detect_size(bytecode):
                    if bytecode[:2] == "0x":
                        return len(bytecode[2:]) / 2
                    elif bytecode[:2] == "0b":
                        return len(bytecode[2:]) / 8
                    else:
                        assert len(set(bytecode)) == 2
                        return len(bytecode) / 8

                df["size"] = df["bytecode"].apply(detect_size)
                df["bytecode"] = df["bytecode"].apply(
                    lambda x: int(x, 16) if "0x" in x else (int(x, 2) if "0b" in x else int(x, 2))
                )
                df["bytecode"] = pd.to_numeric(df["bytecode"])
                df.drop(columns=["rest"], inplace=True)
                return df

            def process_df(df):
                encodings = None
                names = None
                if self.groups:
                    # encodings = re.compile(r"0x[0-9abcdef]+:\s\w+\s#\s([0-9a-fx]+)\s.*").findall(content)
                    # encodings = [f"{enc}" for enc in encodings]
                    encodings = [bin(enc) for enc in df["bytecode"].values]
                if self.sequences:
                    # names = re.compile(r"0x[0-9abcdef]+:\s(\w+)\s#\s[0-9a-fx]+\s.*").findall(content)
                    names = list(df["instr"].values)
                return encodings, names

            log_artifact.uncache()
            encodings = None
            names = None
            if self.groups:
                encodings = []
            if self.sequences:
                names = []
            with pd.read_csv(log_artifact.path, sep=":", names=["pc", "rest"], chunksize=2**22) as reader:  # TODO: expose chunksize
                for chunk in reader:
                    df = transform_df(chunk)

                    encodings_, names_ = process_df(df)
                    # input(">")

                    encodings = encodings_
                    names += names_
            # df = None
            # content = log_artifact.content
            # if self.groups:
            #     encodings = re.compile(r"0x[0-9abcdef]+:\s\w+\s#\s([0-9a-fx]+)\s.*").findall(content)
            #     encodings = [f"0b{enc}" for enc in encodings]
            #     # encodings = [f"{enc}" for enc in encodings]
            # if self.sequences:
            #     names = re.compile(r"0x[0-9abcdef]+:\s(\w+)\s#\s[0-9a-fx]+\s.*").findall(content)
        elif is_ovpsim:
            content = log_artifact.content
            if self.groups:
                encodings = re.compile(r"riscvOVPsim\/cpu',\s0x[0-9abcdef]+\(.*\):\s([0-9abcdef]+)\s+\w+\s+.*").findall(
                    content
                )
                encodings = [f"0x{enc}" for enc in encodings]
            if self.sequences:
                names = re.compile(r"riscvOVPsim\/cpu',\s0x[0-9abcdef]+\(.*\):\s[0-9abcdef]+\s+(\w+)\s+.*").findall(
                    content
                )
        else:
            raise RuntimeError("Uable to determine the used target.")

        def _helper(x, top=100):
            counts = pd.Series(x).value_counts()
            probs = counts / len(x)
            return dict(counts.head(top)), dict(probs.head(top))

        def _gen_csv(label, counts, probs):
            lines = [f"{label},Count,Probability"]
            for x in counts:
                line = f"{x},{counts[x]},{probs[x]:.3f}"
                lines.append(line)
            return "\n".join(lines)

        if self.groups:
            assert is_riscv, "Currently only riscv instrcutions can be analysed by groups"

            def _extract_major_opcode(enc):
                mapping = {
                    0b0010011: "OP-IMM",
                    0b0110111: "LUI",
                    0b0010111: "AUIPC",
                    0b0110011: "OP",
                    0b1101111: "JAL",
                    0b1100111: "JALR",
                    0b1100011: "BRANCH",
                    0b0000011: "LOAD",
                    0b0100011: "STORE",
                    0b0001111: "MISC-MEM",
                    0b1110011: "SYSTEM",
                    0b1000011: "MADD",
                    0b1000111: "MSUB",
                    0b1001011: "MNSUB",
                    0b1001111: "MNADD",
                    0b0000111: "LOAD-FP",
                    0b0100111: "STORE-FP",
                    0b0001011: "custom-0",
                    0b0101011: "custom-1",
                    0b1011011: "custom-2/rv128",
                    0b1111011: "custom-3/rv128",
                    0b1101011: "reserved",
                    0b0101111: "AMO",
                    0b1010011: "OP-FP",
                    0b1010111: "OP-V",
                    0b1110111: "OP-P",
                    0b0011011: "OP-IMM-32",
                    0b0111011: "OP-32",
                }
                enc = int(enc, 0)  # Convert from hexadecimal
                opcode = enc & 0b1111111
                lsbs = opcode & 0b11
                if lsbs == 0b11:
                    major = mapping.get(opcode, "UNKNOWN")
                else:
                    # 16-bit instruction
                    msbs = (enc & 0b1110000000000000) >> 13
                    rvc_mapping = {
                        0b00000: "OP-IMM",
                        0b00001: "OP-IMM",
                        0b00010: "OP-IMM",
                        0b00100: "LOAD",
                        0b00101: "JAL",
                        0b00110: "LOAD-FP",
                        0b01000: "LOAD",
                        0b01001: "OP-IMM",
                        0b01010: "LOAD",
                        0b01100: "LOAD-FP",
                        0b01101: "OP-IMM",
                        0b01110: "LOAD-FP",
                        0b10000: "reserved",
                        0b10001: "MISC-ALU",
                        0b10010: "JALR",
                        0b10100: "STORE-FP",
                        0b10101: "JAL",
                        0b10110: "STORE-FP",
                        0b11000: "STORE",
                        0b11001: "BRANCH",
                        0b11010: "STORE",
                        0b11100: "STORE-FP",
                        0b11101: "BRANCH",
                        0b11110: "STORE-FP",
                    }
                    combined = msbs << 2 | lsbs
                    assert combined in rvc_mapping.keys()
                    return f"{rvc_mapping[combined]} (Compressed)"
                return major

            majors = list(map(_extract_major_opcode, encodings))
            major_counts, major_probs = _helper(majors, top=self.top)
            majors_csv = _gen_csv("Major", major_counts, major_probs)
            artifact = Artifact("analyse_instructions_majors.csv", content=majors_csv, fmt=ArtifactFormat.TEXT)
            if self.to_file:
                ret_artifacts.append(artifact)
            if self.to_df:
                post_df = report.post_df.copy()
                post_df["AnalyseInstructionsMajorsCounts"] = str(major_counts)
                post_df["AnalyseInstructionsMajorsProbs"] = str(major_probs)
                report.post_df = post_df
        if self.sequences:
            max_len = self.seq_depth

            def _get_sublists(lst, length):
                ret = []
                for i in range(len(lst) - length + 1):
                    lst_ = lst[i : i + length]
                    ret.append(";".join(lst_))
                return ret

            for length in range(1, max_len + 1):
                names_ = _get_sublists(names, length)
                counts, probs = _helper(names_, top=self.top)
                sequence_csv = _gen_csv("Sequence", counts, probs)
                artifact = Artifact(
                    f"analyse_instructions_seq{length}.csv", content=sequence_csv, fmt=ArtifactFormat.TEXT
                )
                if self.to_file:
                    ret_artifacts.append(artifact)
                if self.to_df:
                    post_df = report.post_df.copy()
                    post_df[f"AnalyseInstructionsSeq{length}Counts"] = str(counts)
                    post_df[f"AnalyseInstructionsSeq{length}Probs"] = str(probs)
                    report.post_df = post_df
        if self.corev:
            XCVMAC_INSNS = {
                "cv.mac",
                "cv.msu",
                "cv.mulun",
                "cv.mulhhun",
                "cv.mulsn",
                "cv.mulhhsn",
                "cv.mulurn",
                "cv.mulhhurn",
                "cv.mulsrn",
                "cv.mulhhsrn",
                "cv.macun",
                "cv.machhun",
                "cv.macsn",
                "cv.machhsn",
                "cv.macurn",
                "cv.machhurn",
                "cv.macsrn",
                "cv.machhsrn",
            }
            XCVMEM_INSNS = {
                "cv.lb_ri_inc",
                "cv.lbu_ri_inc",
                "cv.lh_ri_inc",
                "cv.lhu_ri_inc",
                "cv.lw_ri_inc",
                "cv.lb_ri_inc",
                "cv.lbu_ri_inc",
                "cv.lh_ri_inc",
                "cv.lhu_ri_inc",
                "cv.lw_ri_inc",
                "cv.lb_rr_inc",
                "cv.lbu_rr_inc",
                "cv.lh_rr_inc",
                "cv.lhu_rr_inc",
                "cv.lw_rr_inc",
                "cv.lb_rr_inc",
                "cv.lbu_rr_inc",
                "cv.lh_rr_inc",
                "cv.lhu_rr_inc",
                "cv.lw_rr_inc",
                "cv.lb_rr",
                "cv.lbu_rr",
                "cv.lh_rr",
                "cv.lhu_rr",
                "cv.lw_rr",
                "cv.sb_ri_inc",
                "cv.sh_ri_inc",
                "cv.sw_ri_inc",
                "cv.sb_ri_inc",
                "cv.sh_ri_inc",
                "cv.sw_ri_inc",
                "cv.sb_rr_inc",
                "cv.sh_rr_inc",
                "cv.sw_rr_inc",
                "cv.sb_rr_inc",
                "cv.sh_rr_inc",
                "cv.sw_rr_inc",
                "cv.sb_rr",
                "cv.sh_rr",
                "cv.sw_rr",
            }
            XCVBI_INSNS = {
                "cv.bneimm",
                "cv.beqimm",
            }
            XCVALU_INSNS = {
                "cv.slet",
                "cv.min",
                "cv.addnr",
                "cv.addunr",
                "cv.addn",
                "cv.maxu",
                "cv.subun",
                "cv.extbz",
                "cv.addun",
                "cv.clip",
                "cv.clipu",
                "cv.subn",
                "cv.max",
                "cv.extbs",
                "cv.abs",
                "cv.addurn",
                "cv.exths",
                "cv.exthz",
                "cv.minu",
                "cv.sletu",
                "cv.suburn",
                "cv.addrn",
                "cv.clipur",
                "cv.subrn",
            }
            XCVBITMANIP_INSNS = {
                "cv.ror",
                "cv.clb",
            }
            XCVSIMD_INSNS = {
                "cv.add.h",
                "cv.add.sc.b",
                "cv.add.sc.h",
                "cv.add.sci.h",
                "cv.and.b",
                "cv.and.h",
                "cv.and.sc.h",
                "cv.and.sci.h",
                "cv.cmpeq.sc.h",
                "cv.cmpge.sci.h",
                "cv.cmpgtu.h",
                "cv.cmplt.sci.h",
                "cv.cmpltu.sci.b",
                "cv.cmpne.sc.h",
                "cv.cmpne.sci.b",
                "cv.extract.b",
                "cv.extract.h",
                "cv.extractu.b",
                "cv.extractu.h",
                "cv.insert.h",
                "cv.max.h",
                "cv.max.sci.h",
                "cv.maxu.h",
                "cv.or.b",
                "cv.or.h",
                "cv.pack",
                "cv.packhi.b",
                "cv.packlo.b",
                "cv.shuffle2.b",
                "cv.shuffle2.h",
                "cv.shufflei0.sci.b",
                "cv.sll.sci.h",
                "cv.sra.h",
                "cv.sra.sci.h",
                "cv.srl.h",
                "cv.srl.sci.h",
                "cv.sub.b",
                "cv.sub.h",
                "cv.xor.b",
                "cv.xor.sci.b",
                "cv.add.sci.b",
                "cv.cmpeq.b",
                "cv.cmpgtu.sc.h",
                "cv.cmpleu.sc.h",
                "cv.sdotup.h",
                "cv.sdotup.b",
                "cv.shuffle.sci.h",
                "cv.xor.sc.b",
                "cv.xor.sc.h",
                "cv.sdotsp.h",
                "cv.cmpeq.sci.b",
                "cv.and.sci.b",
                "cv.dotsp.h",
                "cv.dotsp.b",
                "cv.sdotsp.b",
                "cv.add.b",
                "cv.dotup.sci.b",
            }
            XCVHWLP_INSNS = {
                "cv.count",
                "cv.counti",
                "cv.start",
                "cv.starti",
                "cv.end",
                "cv.endi",
                "cv.setup",
                "cv.setupi",
            }

            def apply_mapping(x):
                x = x.replace("cv_", "cv.")
                x = x.replace("_sc", ".sc")
                x = x.replace("_b", ".b")
                x = x.replace("_h", ".h")
                if x in XCVMAC_INSNS:
                    return "XCVMac"
                elif x in XCVMEM_INSNS:
                    return "XCVMem"
                elif x in XCVALU_INSNS:
                    return "XCVAlu"
                elif x in XCVBITMANIP_INSNS:
                    return "XCVBitmanip"
                elif x in XCVBI_INSNS:
                    return "XCVBi"
                elif x in XCVSIMD_INSNS:
                    return "XCVSimd"
                elif x in XCVHWLP_INSNS:
                    return "XCVHwlp"
                elif "cv." in x:
                    return "XCV?"
                else:
                    return "Other"

            names_ = list(map(apply_mapping, names))
            cv_ext_counts, cv_ext_probs = _helper(names_, top=self.top)
            corev_csv = _gen_csv("Set", cv_ext_counts, cv_ext_probs)
            artifact = Artifact("analyse_instructions_corev.csv", content=corev_csv, fmt=ArtifactFormat.TEXT)
            if self.to_file:
                ret_artifacts.append(artifact)
            if self.to_df:
                post_df = report.post_df.copy()
                post_df["CoreVSetCounts"] = str(cv_ext_counts)
                post_df["CoreVSetProbs"] = str(cv_ext_probs)
                report.post_df = post_df
        assert self.to_file or self.to_df, "Either to_file or to_df have to be true"
        return ret_artifacts


class CompareRowsPostprocess(SessionPostprocess):
    """TODO"""

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "to_compare": None,
        "group_by": None,
        "baseline": 0,
        "percent": False,
        "invert": False,
        "substract": False,
    }

    def __init__(self, features=None, config=None):
        super().__init__("compare_rows", features=features, config=config)

    @property
    def to_compare(self):
        """Get to_compare property."""
        value = self.config["to_compare"]
        return str2list(value, allow_none=True)

    @property
    def group_by(self):
        """Get group_by property."""
        value = self.config["group_by"]
        return str2list(value, allow_none=True)

    @property
    def baseline(self):
        """Get baseline property."""
        value = self.config["baseline"]
        return int(value)

    @property
    def percent(self):
        """Get percent property."""
        value = self.config["percent"]
        return str2bool(value)

    @property
    def invert(self):
        """Get invert property."""
        value = self.config["invert"]
        return str2bool(value)

    @property
    def substract(self):
        """Get substract property."""
        value = self.config["substract"]
        return str2bool(value)

    def post_session(self, report):
        """Called at the end of a session."""
        pre_df = report.pre_df
        main_df = report.main_df  # metrics
        post_df = report.post_df
        group_by = self.group_by
        if group_by is None:
            group_by = [x for x in pre_df.columns if x not in ["Run", "Sub"]]
        assert isinstance(group_by, list)
        assert all(col in list(pre_df.columns) + list(post_df.columns) for col in group_by), "Cols mssing in df"
        to_compare = self.to_compare
        if to_compare is None:
            to_compare = list(main_df.columns)
        assert isinstance(to_compare, list)
        assert all(col in main_df.columns for col in to_compare)
        full_df = pd.concat([pre_df, main_df, post_df], axis=1)
        grouped = full_df.groupby(group_by, axis=0, group_keys=False, dropna=False)
        new_df = pd.DataFrame()
        for col in to_compare:

            def f(df):
                assert self.baseline < len(df), "Index of group baseline out of bounds"
                ret = df / df.iloc[self.baseline]
                if self.substract:
                    ret = ret - 1
                if self.invert:
                    ret = 1 / ret
                if self.percent:
                    ret = ret * 100.0
                return ret

            filtered_col = grouped[col]
            first = filtered_col.apply(f).reset_index()
            first_col = first[col]
            new = first_col
            new_name = f"{col} (rel.)"
            new_df[new_name] = new
        main_df = pd.concat([main_df, new_df], axis=1)
        report.main_df = main_df


class AnalyseDumpPostprocess(RunPostprocess):
    """Counting static instructions."""

    DEFAULTS = {
        **RunPostprocess.DEFAULTS,
        "to_df": False,
        "to_file": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("analyse_dump", features=features, config=config)

    @property
    def to_df(self):
        """Get to_df property."""
        value = self.config["to_df"]
        return str2bool(value)

    @property
    def to_file(self):
        """Get to_file property."""
        value = self.config["to_file"]
        return str2bool(value)

    def post_run(self, report, artifacts):
        """Called at the end of a run."""
        platform = report.pre_df["Platform"]
        if (platform != "mlif").any():
            return []
        ret_artifacts = []
        dump_artifact = lookup_artifacts(
            artifacts, name="generic_mlonmcu.dump", fmt=ArtifactFormat.TEXT, first_only=True
        )
        assert len(dump_artifact) == 1, "To use analyse_dump postprocess, please set mlif.enable_asmdump=1"
        dump_artifact = dump_artifact[0]
        is_llvm = "llvm" in dump_artifact.flags
        assert is_llvm, "Non-llvm objdump currently unsupported"
        content = dump_artifact.content
        lines = content.split("\n")
        counts = {}
        total = 0
        for line in lines:
            splitted = line.split("\t")
            if len(splitted) != 3:
                continue
            insn = splitted[1]
            args = splitted[2]
            if "cv." in insn:
                if "(" in args and ")" in args:
                    m = re.compile(r"(.*)\((.*)\)").match(args)
                    if m:
                        g = m.groups()
                        assert len(g) == 2
                        offset, base = g
                        fmt = "ri"
                        try:
                            offset = int(offset)
                        except ValueError:
                            fmt = "rr"
                        insn += f"_{fmt}"
                        if "!" in base:
                            insn += "_inc"
            if insn in counts:
                counts[insn] += 1
            else:
                counts[insn] = 1
            total += 1
        counts_csv = "Instruction,Count,Probability\n"
        for insn, count in sorted(counts.items(), key=lambda item: item[1]):
            counts_csv += f"{insn},{count},{count/total:.4f}\n"
        artifact = Artifact("dump_counts.csv", content=counts_csv, fmt=ArtifactFormat.TEXT)
        if self.to_file:
            ret_artifacts.append(artifact)
        if self.to_df:
            post_df = report.post_df.copy()
            post_df["DumpCounts"] = str(counts)
            report.post_df = post_df
        assert self.to_file or self.to_df, "Either to_file or to_df have to be true"
        return ret_artifacts


class AnalyseCoreVCountsPostprocess(RunPostprocess):
    """Counting static instructions."""

    DEFAULTS = {
        **RunPostprocess.DEFAULTS,
        "to_df": False,
        "to_file": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("analyse_corev_counts", features=features, config=config)

    @property
    def to_df(self):
        """Get to_df property."""
        value = self.config["to_df"]
        return str2bool(value)

    @property
    def to_file(self):
        """Get to_file property."""
        value = self.config["to_file"]
        return str2bool(value)

    def post_run(self, report, artifacts):
        """Called at the end of a run."""
        ret_artifacts = []
        count_artifact = lookup_artifacts(artifacts, name="dump_counts.csv", fmt=ArtifactFormat.TEXT, first_only=True)
        assert len(count_artifact) == 1, "To use analyse_corev_counts postprocess, analyse_dump needs to run first."
        count_artifact = count_artifact[0]
        content = count_artifact.content

        lines = content.split("\n")

        XCVMAC_INSNS = {
            "cv.mac",
            "cv.msu",
            "cv.mulun",
            "cv.mulhhun",
            "cv.mulsn",
            "cv.mulhhsn",
            "cv.mulurn",
            "cv.mulhhurn",
            "cv.mulsrn",
            "cv.mulhhsrn",
            "cv.macun",
            "cv.machhun",
            "cv.macsn",
            "cv.machhsn",
            "cv.macurn",
            "cv.machhurn",
            "cv.macsrn",
            "cv.machhsrn",
        }
        XCVMEM_INSNS = {
            "cv.lb_ri_inc",
            "cv.lbu_ri_inc",
            "cv.lh_ri_inc",
            "cv.lhu_ri_inc",
            "cv.lw_ri_inc",
            "cv.lb_ri_inc",
            "cv.lbu_ri_inc",
            "cv.lh_ri_inc",
            "cv.lhu_ri_inc",
            "cv.lw_ri_inc",
            "cv.lb_rr_inc",
            "cv.lbu_rr_inc",
            "cv.lh_rr_inc",
            "cv.lhu_rr_inc",
            "cv.lw_rr_inc",
            "cv.lb_rr_inc",
            "cv.lbu_rr_inc",
            "cv.lh_rr_inc",
            "cv.lhu_rr_inc",
            "cv.lw_rr_inc",
            "cv.lb_rr",
            "cv.lbu_rr",
            "cv.lh_rr",
            "cv.lhu_rr",
            "cv.lw_rr",
            "cv.sb_ri_inc",
            "cv.sh_ri_inc",
            "cv.sw_ri_inc",
            "cv.sb_ri_inc",
            "cv.sh_ri_inc",
            "cv.sw_ri_inc",
            "cv.sb_rr_inc",
            "cv.sh_rr_inc",
            "cv.sw_rr_inc",
            "cv.sb_rr_inc",
            "cv.sh_rr_inc",
            "cv.sw_rr_inc",
            "cv.sb_rr",
            "cv.sh_rr",
            "cv.sw_rr",
        }
        XCVBI_INSNS = {
            "cv.bneimm",
            "cv.beqimm",
        }
        XCVALU_INSNS = {
            "cv.slet",
            "cv.min",
            "cv.addnr",
            "cv.addunr",
            "cv.addn",
            "cv.maxu",
            "cv.subun",
            "cv.extbz",
            "cv.addun",
            "cv.clip",
            "cv.clipu",
            "cv.subn",
            "cv.max",
            "cv.extbs",
            "cv.abs",
            "cv.addurn",
            "cv.exths",
            "cv.exthz",
            "cv.minu",
            "cv.sletu",
            "cv.suburn",
            "cv.addrn",
            "cv.clipur",
            "cv.subrn",
        }
        XCVBITMANIP_INSNS = {
            "cv.ror",
            "cv.clb",
        }
        XCVSIMD_INSNS = {
            "cv.add.h",
            "cv.add.sc.b",
            "cv.add.sc.h",
            "cv.add.sci.h",
            "cv.and.b",
            "cv.and.h",
            "cv.and.sc.h",
            "cv.and.sci.h",
            "cv.cmpeq.sc.h",
            "cv.cmpge.sci.h",
            "cv.cmpgtu.h",
            "cv.cmplt.sci.h",
            "cv.cmpltu.sci.b",
            "cv.cmpne.sc.h",
            "cv.cmpne.sci.b",
            "cv.extract.b",
            "cv.extract.h",
            "cv.extractu.b",
            "cv.extractu.h",
            "cv.insert.h",
            "cv.max.h",
            "cv.max.sci.h",
            "cv.maxu.h",
            "cv.or.b",
            "cv.or.h",
            "cv.pack",
            "cv.packhi.b",
            "cv.packlo.b",
            "cv.shuffle2.b",
            "cv.shuffle2.h",
            "cv.shufflei0.sci.b",
            "cv.sll.sci.h",
            "cv.sra.h",
            "cv.sra.sci.h",
            "cv.srl.h",
            "cv.srl.sci.h",
            "cv.sub.b",
            "cv.sub.h",
            "cv.xor.b",
            "cv.xor.sci.b",
            "cv.add.sci.b",
            "cv.cmpeq.b",
            "cv.cmpgtu.sc.h",
            "cv.cmpleu.sc.h",
            "cv.sdotup.h",
            "cv.sdotup.b",
            "cv.shuffle.sci.h",
            "cv.xor.sc.b",
            "cv.xor.sc.h",
            "cv.sdotsp.h",
            "cv.cmpeq.sci.b",
            "cv.and.sci.b",
            "cv.dotsp.h",
            "cv.dotsp.b",
            "cv.sdotsp.b",
            "cv.add.b",
            "cv.dotup.sci.b",
        }
        XCVHWLP_INSNS = {
            "cv.count",
            "cv.counti",
            "cv.start",
            "cv.starti",
            "cv.end",
            "cv.endi",
            "cv.setup",
            "cv.setupi",
        }

        unknowns = []
        cv_ext_totals = {
            "XCVMac": len(XCVMAC_INSNS),
            "XCVMem": len(XCVMEM_INSNS),
            "XCVBi": len(XCVBI_INSNS),
            "XCVAlu": len(XCVALU_INSNS),
            "XCVBitmanip": len(XCVBITMANIP_INSNS),
            "XCVSimd": len(XCVSIMD_INSNS),
            "XCVHwlp": len(XCVHWLP_INSNS),
            "Unknown": 0,
        }
        cv_ext_counts = {
            "XCVMac": 0,
            "XCVMem": 0,
            "XCVBi": 0,
            "XCVAlu": 0,
            "XCVBitmanip": 0,
            "XCVSimd": 0,
            "XCVHwlp": 0,
            "Unknown": 0,
        }
        cv_ext_unique_counts = {
            "XCVMac": 0,
            "XCVMem": 0,
            "XCVBi": 0,
            "XCVAlu": 0,
            "XCVBitmanip": 0,
            "XCVSimd": 0,
            "XCVHwlp": 0,
            "Unknown": 0,
        }
        total_counts = 0
        cv_counts_csv = "Instruction,Count,Probability\n"
        cv_counts = {}
        for line in lines[1:]:
            if "cv." not in line:
                continue
            cv_counts_csv += f"{line}\n"
            splitted = line.split(",")
            assert len(splitted) == 3
            insn = splitted[0]
            count = int(splitted[1])
            cv_counts[insn] = count
            total_counts += count
            if insn in XCVMAC_INSNS:
                cv_ext_counts["XCVMac"] += count
                cv_ext_unique_counts["XCVMac"] += 1
            elif insn in XCVMEM_INSNS:
                cv_ext_counts["XCVMem"] += count
                cv_ext_unique_counts["XCVMem"] += 1
            elif insn in XCVBI_INSNS:
                cv_ext_counts["XCVBi"] += count
                cv_ext_unique_counts["XCVBi"] += 1
            elif insn in XCVALU_INSNS:
                cv_ext_counts["XCVAlu"] += count
                cv_ext_unique_counts["XCVAlu"] += 1
            elif insn in XCVBITMANIP_INSNS:
                cv_ext_counts["XCVBitmanip"] += count
                cv_ext_unique_counts["XCVBitmanip"] += 1
            elif insn in XCVSIMD_INSNS:
                cv_ext_counts["XCVSimd"] += count
                cv_ext_unique_counts["XCVSimd"] += 1
            elif insn in XCVHWLP_INSNS:
                cv_ext_counts["XCVHwlp"] += count
                cv_ext_unique_counts["XCVHwlp"] += 1
            else:
                cv_ext_counts["Unknown"] += count
                cv_ext_unique_counts["Unknown"] += 1
                if insn not in unknowns:
                    unknowns.append(insn)
        cv_ext_totals["Unknown"] = len(unknowns)
        cv_ext_counts_csv = "Set,Count,Probability\n"
        for ext, count in sorted(cv_ext_counts.items(), key=lambda item: item[1]):
            if count == 0:
                continue
            cv_ext_counts_csv += f"{ext},{count},{count/total_counts}\n"
        cv_ext_unique_counts_csv = "Set,Used,Utilization\n"
        for ext, used in sorted(cv_ext_unique_counts.items(), key=lambda item: item[1]):
            if used == 0:
                continue
            rel = used / cv_ext_totals[ext]
            cv_ext_unique_counts_csv += f"{ext},{used},{rel:.4f}\n"
        used = sum(cv_ext_unique_counts.values())
        totals = sum(cv_ext_totals.values())
        rel = used / totals
        cv_ext_unique_counts_csv += f"XCVTotal,{used},{rel:.4f}\n"

        cv_counts_artifact = Artifact("cv_counts.csv", content=cv_counts_csv, fmt=ArtifactFormat.TEXT)
        cv_ext_counts_artifact = Artifact("cv_ext_counts.csv", content=cv_ext_counts_csv, fmt=ArtifactFormat.TEXT)
        cv_ext_unique_counts_artifact = Artifact(
            "cv_ext_unique_counts.csv", content=cv_ext_unique_counts_csv, fmt=ArtifactFormat.TEXT
        )
        if len(unknowns) > 0:
            logger.warning("Unknown instructions found: %s", unknowns)
            cv_ext_unknowns_artifact = Artifact(
                "cv_ext_unknowns.csv", content="\n".join(unknowns), fmt=ArtifactFormat.TEXT
            )
            if self.to_file:
                ret_artifacts.append(cv_ext_unknowns_artifact)
            # TODO: logging

        if self.to_file:
            ret_artifacts.append(cv_counts_artifact)
            ret_artifacts.append(cv_ext_counts_artifact)
            ret_artifacts.append(cv_ext_unique_counts_artifact)
        if self.to_df:
            post_df = report.post_df.copy()
            post_df["XCVCounts"] = str(cv_counts)
            post_df["XCVExtCounts"] = str(cv_ext_counts)
            post_df["XCVExtUniqueCounts"] = str(cv_ext_unique_counts)
            report.post_df = post_df
        assert self.to_file or self.to_df, "Either to_file or to_df have to be true"
        return ret_artifacts


class ValidateOutputsPostprocess(RunPostprocess):
    """Postprocess for comparing model outputs with golden reference."""

    DEFAULTS = {
        **RunPostprocess.DEFAULTS,
        "report": False,
        "validate_metrics": "topk(n=1);topk(n=2)",
        "validate_range": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("validate_outputs", features=features, config=config)

    @property
    def validate_metrics(self):
        """Get validate_metrics property."""
        value = self.config["validate_metrics"]
        return value

    @property
    def report(self):
        """Get report property."""
        value = self.config["report"]
        return str2bool(value)

    @property
    def validate_range(self):
        """Get validate_range property."""
        value = self.config["validate_range"]
        return str2bool(value)

    def post_run(self, report, artifacts):
        """Called at the end of a run."""
        model_info_artifact = lookup_artifacts(artifacts, name="model_info.yml", first_only=True)
        assert len(model_info_artifact) == 1, "Could not find artifact: model_info.yml"
        model_info_artifact = model_info_artifact[0]
        import yaml

        model_info_data = yaml.safe_load(model_info_artifact.content)
        if len(model_info_data["output_names"]) > 1:
            raise NotImplementedError("Multi-outputs not yet supported.")
        outputs_ref_artifact = lookup_artifacts(artifacts, name="outputs_ref.npy", first_only=True)
        assert len(outputs_ref_artifact) == 1, "Could not find artifact: outputs_ref.npy"
        outputs_ref_artifact = outputs_ref_artifact[0]
        import numpy as np

        outputs_ref = np.load(outputs_ref_artifact.path, allow_pickle=True)
        # import copy
        # outputs = copy.deepcopy(outputs_ref)
        # outputs[1][list(outputs[1].keys())[0]][0] = 42
        outputs_artifact = lookup_artifacts(artifacts, name="outputs.npy", first_only=True)
        assert len(outputs_artifact) == 1, "Could not find artifact: outputs.npy"
        outputs_artifact = outputs_artifact[0]
        outputs = np.load(outputs_artifact.path, allow_pickle=True)
        in_data = None
        # compared = 0
        # matching = 0
        # missing = 0
        # metrics = {
        #     "allclose(atol=0.0,rtol=0.0)": None,
        #     "allclose(atol=0.05,rtol=0.05)": None,
        #     "allclose(atol=0.1,rtol=0.1)": None,
        #     "topk(n=1)": None,
        #     "topk(n=2)": None,
        #     "topk(n=inf)": None,
        #     "toy": None,
        #     "mse(thr=0.1)": None,
        #     "mse(thr=0.05)": None,
        #     "mse(thr=0.01)": None,
        #     "+-1": None,
        # }
        validate_metrics_str = self.validate_metrics
        validate_metrics = parse_validate_metrics(validate_metrics_str)
        for i, output_ref in enumerate(outputs_ref):
            if i >= len(outputs):
                logger.warning("Missing output sample")
                # missing += 1
                break
            output = outputs[i]
            ii = 0
            for out_name, out_ref_data in output_ref.items():
                if out_name in output:
                    out_data = output[out_name]
                elif ii < len(output):
                    if isinstance(output, dict):
                        # fallback for custom name-based npy dict
                        out_data = list(output.values())[ii]
                    else:  # fallback for index-based npy array
                        assert isinstance(output, (list, np.array)), "expected dict, list or np.array type"
                        out_data = output[ii]
                else:
                    RuntimeError(f"Output not found: {out_name}")
                # optional dequantize
                # print("out_data_before_quant", out_data)
                # print("sum(out_data_before_quant", np.sum(out_data))

                quant = model_info_data.get("output_quant_details", None)
                rng = model_info_data.get("output_ranges", None)
                if quant:

                    def ref_quant_helper(quant, data):  # TODO: move somewhere else
                        if quant is None:
                            return data
                        quant_scale, quant_zero_point, quant_dtype, quant_range = quant
                        if quant_dtype is None or data.dtype.name == quant_dtype:
                            return data
                        assert data.dtype.name in ["float32"], "Quantization only supported for float32 input"
                        assert quant_dtype in ["int8"], "Quantization only supported for int8 output"
                        if quant_range and self.validate_range:
                            assert len(quant_range) == 2, "Range should be a tuple (lower, upper)"
                            lower, upper = quant_range
                            # print("quant_range", quant_range)
                            # print("np.min(data)", np.min(data))
                            # print("np.max(data)", np.max(data))
                            assert lower <= upper
                            assert np.min(data) >= lower and np.max(data) <= upper, "Range missmatch"

                        return np.around((data / quant_scale) + quant_zero_point).astype("int8")

                    def dequant_helper(quant, data):  # TODO: move somewhere else
                        if quant is None:
                            return data
                        quant_scale, quant_zero_point, quant_dtype, quant_range = quant
                        if quant_dtype is None or data.dtype.name == quant_dtype:
                            return data
                        assert data.dtype.name in ["int8"], "Dequantization only supported for int8 input"
                        assert quant_dtype in ["float32"], "Dequantization only supported for float32 output"
                        ret = (data.astype("float32") - quant_zero_point) * quant_scale
                        if quant_range and self.validate_range:
                            assert len(quant_range) == 2, "Range should be a tuple (lower, upper)"
                            # print("quant_range", quant_range)
                            # print("np.min(ret)", np.min(ret))
                            # print("np.max(ret)", np.max(ret))
                            lower, upper = quant_range
                            assert lower <= upper
                            assert np.min(ret) >= lower and np.max(ret) <= upper, "Range missmatch"
                        return ret

                    assert ii < len(rng)
                    rng_ = rng[ii]
                    if rng_ and self.validate_range:
                        assert len(rng_) == 2, "Range should be a tuple (lower, upper)"
                        lower, upper = rng_
                        assert lower <= upper
                        # print("rng_", rng_)
                        # print("np.min(out_data)", np.min(out_data))
                        # print("np.max(out_data)", np.max(out_data))
                        assert np.min(out_data) >= lower and np.max(out_data) <= upper, "Range missmatch"
                    assert ii < len(quant)
                    quant_ = quant[ii]
                    if quant_ is not None:
                        out_ref_data_quant = ref_quant_helper(quant_, out_ref_data)
                        for vm in validate_metrics:
                            vm.process(out_data, out_ref_data_quant, in_data=in_data, quant=True)
                        out_data = dequant_helper(quant_, out_data)
                # print("out_data", out_data)
                # print("sum(out_data)", np.sum(out_data))
                # print("out_ref_data", out_ref_data)
                # print("sum(out_ref_data)", np.sum(out_ref_data))
                # input("TIAW")
                assert out_data.dtype == out_ref_data.dtype, "dtype missmatch"
                assert out_data.shape == out_ref_data.shape, "shape missmatch"

                for vm in validate_metrics:
                    vm.process(out_data, out_ref_data, in_data=in_data, quant=False)
                ii += 1
        if self.report:
            raise NotImplementedError
        for vm in validate_metrics:
            res = vm.get_summary()
            report.post_df[f"{vm.name}"] = res
        return []


class ValidateLabelsPostprocess(RunPostprocess):
    """Postprocess for comparing model outputs with golden reference."""

    DEFAULTS = {
        **RunPostprocess.DEFAULTS,
        "report": False,
        "classify_metrics": "topk_label(n=1);topk_label(n=2)",
    }

    def __init__(self, features=None, config=None):
        super().__init__("validate_labels", features=features, config=config)

    @property
    def classify_metrics(self):
        """Get classify_metrics property."""
        value = self.config["classify_metrics"]
        return value

    @property
    def report(self):
        """Get report property."""
        value = self.config["report"]
        return str2bool(value)

    def post_run(self, report, artifacts):
        """Called at the end of a run."""
        model_info_artifact = lookup_artifacts(artifacts, name="model_info.yml", first_only=True)
        assert len(model_info_artifact) == 1, "Could not find artifact: model_info.yml"
        model_info_artifact = model_info_artifact[0]
        import yaml

        model_info_data = yaml.safe_load(model_info_artifact.content)
        if len(model_info_data["output_names"]) > 1:
            raise NotImplementedError("Multi-outputs not yet supported.")
        labels_ref_artifact = lookup_artifacts(artifacts, name="labels_ref.npy", first_only=True)
        assert (
            len(labels_ref_artifact) == 1
        ), "Could not find artifact: labels_ref.npy (Run classify_labels postprocess first!)"
        labels_ref_artifact = labels_ref_artifact[0]
        import numpy as np

        labels_ref = np.load(labels_ref_artifact.path, allow_pickle=True)
        outputs_artifact = lookup_artifacts(artifacts, name="outputs.npy", first_only=True)
        assert len(outputs_artifact) == 1, "Could not find artifact: outputs.npy"
        outputs_artifact = outputs_artifact[0]
        outputs = np.load(outputs_artifact.path, allow_pickle=True)
        # missing = 0
        classify_metrics_str = self.classify_metrics
        classify_metrics = parse_classify_metrics(classify_metrics_str)
        for i, output in enumerate(outputs):
            if isinstance(output, dict):  # name based lookup
                pass
            else:  # index based lookup
                assert isinstance(output, (list, np.array)), "expected dict, list or np.array"
                output_names = model_info_data["output_names"]
                assert len(output) == len(output_names)
                output = {output_names[idx]: out for idx, out in enumerate(output)}
            assert len(output) == 1, "Only supporting single-output models"
            out_data = output[list(output.keys())[0]]
            # print("out_data", out_data)
            assert i < len(labels_ref), "Missing reference labels"
            label_ref = labels_ref[i]
            # print("label_ref", label_ref)
            for cm in classify_metrics:
                cm.process(out_data, label_ref, quant=False)
        if self.report:
            raise NotImplementedError
        for cm in classify_metrics:
            res = cm.get_summary()
            report.post_df[f"{cm.name}"] = res
        return []


class ExportOutputsPostprocess(RunPostprocess):
    """Postprocess for writing model outputs to a directory."""

    DEFAULTS = {
        **RunPostprocess.DEFAULTS,
        "dest": None,  # if none: export as artifact
        "use_ref": False,
        "skip_dequant": False,
        "fmt": "bin",
        "archive_fmt": None,
    }

    def __init__(self, features=None, config=None):
        super().__init__("export_outputs", features=features, config=config)

    @property
    def dest(self):
        """Get dest property."""
        value = self.config["dest"]
        if value is not None:
            if not isinstance(value, Path):
                assert isinstance(value, str)
                value = Path(value)
        return value

    @property
    def use_ref(self):
        """Get use_ref property."""
        value = self.config["use_ref"]
        return str2bool(value)

    @property
    def skip_dequant(self):
        """Get skip_dequant property."""
        value = self.config["skip_dequant"]
        return str2bool(value)

    @property
    def fmt(self):
        """Get fmt property."""
        return self.config["fmt"]

    @property
    def archive_fmt(self):
        """Get archive_fmt property."""
        return self.config["archive_fmt"]

    def post_run(self, report, artifacts):
        """Called at the end of a run."""
        model_info_artifact = lookup_artifacts(artifacts, name="model_info.yml", first_only=True)
        assert len(model_info_artifact) == 1, "Could not find artifact: model_info.yml"
        model_info_artifact = model_info_artifact[0]
        import yaml

        model_info_data = yaml.safe_load(model_info_artifact.content)
        # print("model_info_data", model_info_data)
        if len(model_info_data["output_names"]) > 1:
            raise NotImplementedError("Multi-outputs not yet supported.")
        if self.use_ref:
            outputs_ref_artifact = lookup_artifacts(artifacts, name="outputs_ref.npy", first_only=True)
            assert len(outputs_ref_artifact) == 1, "Could not find artifact: outputs_ref.npy"
            outputs_ref_artifact = outputs_ref_artifact[0]
            outputs_ref = np.load(outputs_ref_artifact.path, allow_pickle=True)
            outputs = outputs_ref
        else:
            outputs_artifact = lookup_artifacts(artifacts, name="outputs.npy", first_only=True)
            assert len(outputs_artifact) == 1, "Could not find artifact: outputs.npy"
            outputs_artifact = outputs_artifact[0]
            outputs = np.load(outputs_artifact.path, allow_pickle=True)
        if self.dest is None:
            temp_dir = tempfile.TemporaryDirectory()
            dest_ = Path(temp_dir.name)
        else:
            temp_dir = None
            assert self.dest.is_dir(), f"Not a directory: {self.dest}"
            dest_ = self.dest
        assert self.fmt in ["bin", "npy"], f"Invalid format: {self.fmt}"
        filenames = []
        for i, output in enumerate(outputs):
            if isinstance(output, dict):  # name based lookup
                pass
            else:  # index based lookup
                assert isinstance(output, (list, np.array)), "expected dict, list or np.array"
                output_names = model_info_data["output_names"]
                assert len(output) == len(output_names)
                output = {output_names[idx]: out for idx, out in enumerate(output)}
            quant = model_info_data.get("output_quant_details", None)
            if quant and not self.skip_dequant:

                def dequant_helper(quant, data):
                    if quant is None:
                        return data
                    quant_scale, quant_zero_point, quant_dtype, quant_range = quant
                    if quant_dtype is None or data.dtype.name == quant_dtype:
                        return data
                    assert data.dtype.name in ["int8"], "Dequantization only supported for int8 input"
                    assert quant_dtype in ["float32"], "Dequantization only supported for float32 output"
                    return (data.astype("float32") - quant_zero_point) * quant_scale

                output = {
                    out_name: dequant_helper(quant[j], output[out_name]) for j, out_name in enumerate(output.keys())
                }
            if self.fmt == "npy":
                raise NotImplementedError("npy export")
            elif self.fmt == "bin":
                assert len(output.keys()) == 1, "Multi-outputs not supported"
                output_data = list(output.values())[0]
                data = output_data.tobytes(order="C")
                file_name = f"{i}.bin"
                file_dest = dest_ / file_name
                filenames.append(file_dest)
                with open(file_dest, "wb") as f:
                    f.write(data)
            else:
                assert False, f"fmt not supported: {self.fmt}"
        artifacts = []
        archive_fmt = self.archive_fmt
        create_artifact = self.dest is None or archive_fmt is not None
        if create_artifact:
            if archive_fmt is None:
                assert self.dest is None
                archive_fmt = "tar.gz"  # Default fallback
            assert archive_fmt in ["tar.xz", "tar.gz", "zip"]
            archive_name = f"output_data.{archive_fmt}"
            archive_path = f"{dest_}.{archive_fmt}"
            if archive_fmt == "tar.gz":
                import tarfile

                with tarfile.open(archive_path, "w:gz") as tar:
                    for filename in filenames:
                        tar.add(filename, arcname=filename.name)
            else:
                raise NotImplementedError(f"archive_fmt={archive_fmt}")
            with open(archive_path, "rb") as f:
                raw = f.read()
            artifact = Artifact(archive_name, raw=raw, fmt=ArtifactFormat.BIN)
            artifacts.append(artifact)
        if temp_dir:
            temp_dir.cleanup()
        return artifacts
