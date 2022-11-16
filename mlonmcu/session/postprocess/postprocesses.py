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

import pandas as pd

from mlonmcu.artifact import Artifact, ArtifactFormat, lookup_artifacts
from mlonmcu.config import str2dict, str2bool
from mlonmcu.logging import get_logger

from .postprocess import SessionPostprocess, RunPostprocess

logger = get_logger()


def match_rows(df, cols):
    """Helper function to group similar rows in a dataframe."""
    groups = df.astype(str).groupby(cols).apply(lambda x: tuple(x.index)).tolist()
    return groups


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
            return ast.literal_eval(cfg)
        return cfg

    @property
    def drop(self):
        """Get drop property."""
        cfg = self.config["drop"]
        if isinstance(cfg, str):
            return ast.literal_eval(cfg)
        return cfg

    @property
    def drop_nan(self):
        """Get drop_nan property."""
        value = self.config["drop_nan"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def drop_empty(self):
        """Get drop_empty property."""
        value = self.config["drop_empty"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def drop_const(self):
        """Get drop_const property."""
        value = self.config["drop_const"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

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


class RenameColumnsPostprocess(SessionPostprocess):  # RunPostprocess?
    """Postprocess which can rename columns based on a provided mapping."""

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "mapping": {},
    }

    def __init__(self, features=None, config=None):
        super().__init__("rename_cols", features=features, config=config)

    @property
    def mapping(self):
        value = self.config["mapping"]
        if not isinstance(value, dict):
            return str2dict(value)
        return value

    def post_session(self, report):
        """Called at the end of a session."""
        report.pre_df = report.pre_df.rename(columns=self.mapping)
        report.main_df = report.main_df.rename(columns=self.mapping)
        report.post_df = report.post_df.rename(columns=self.mapping)


class Features2ColumnsPostprocess(SessionPostprocess):  # RunPostprocess?
    """Postprocess which can be used to transform (explode) the 'Features' Column
    in a dataframe for easier filtering."""

    def __init__(self, features=None, config=None):
        super().__init__("features2cols", features=features, config=config)

    def post_session(self, report):
        df = report.post_df
        if "Features" not in df.columns:
            return
        feature_df = pd.concat(
            [
                df["Features"].apply(lambda x: pd.Series({"feature_" + feature_name: feature_name in x}))
                for feature_name in list(set(df["Features"].sum()))
            ],
            axis=1,
        )
        tmp_df = df.drop(columns=["Features"])
        new_df = pd.concat([tmp_df, feature_df], axis=1)
        report.post_df = new_df


class Config2ColumnsPostprocess(SessionPostprocess):  # RunPostprocess?
    """Postprocess which can be used to transform (explode) the 'Config' Column in a dataframe for easier filtering."""

    def __init__(self, features=None, config=None):
        super().__init__("config2cols", features=features, config=config)

    def post_session(self, report):
        """Called at the end of a session."""
        df = report.post_df
        if "Config" not in df.columns:
            return
        config_df = df["Config"].apply(pd.Series).add_prefix("config_")
        tmp_df = df.drop(columns=["Config"])
        new_df = pd.concat([tmp_df, config_df], axis=1)
        report.post_df = new_df


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
            matches = lookup_artifacts(artifacts, name=filename, first_only=True)
            if not matches:
                report.main_df[colname] = ""
                return
            if matches[0].fmt != ArtifactFormat.TEXT:
                raise RuntimeError("Can only put text into report columns")
            report.main_df[colname] = matches[0].content


class AnalyseInstructionsPostprocess(RunPostprocess):
    """Counting specific types of instructions."""

    DEFAULTS = {**RunPostprocess.DEFAULTS, "groups": True, "sequences": True, "top": 10}

    def __init__(self, features=None, config=None):
        super().__init__("analyse_instructions", features=features, config=config)

    @property
    def groups(self):
        """Get groups property."""
        value = self.config["groups"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def sequences(self):
        """get sequences property."""
        value = self.config["sequences"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def top(self):
        """get sequences property."""
        return int(self.config["top"])

    def post_run(self, report, artifacts):
        """Called at the end of a run."""
        ret_artifacts = []
        log_artifact = lookup_artifacts(artifacts, flags=("log_instrs",), fmt=ArtifactFormat.TEXT, first_only=True)
        assert len(log_artifact) == 1, "To use analyse_instructions process, please enable feature log_instrs."
        log_artifact = log_artifact[0]
        is_spike = "spike" in log_artifact.flags
        is_etiss = "etiss_pulpino" in log_artifact.flags
        is_ovpsim = "ovpsim" in log_artifact.flags
        is_riscv = is_spike or is_etiss or is_ovpsim
        if is_spike:
            content = log_artifact.content
            if self.groups:
                encodings = re.compile(r"\((0x[0-9abcdef]+)\)").findall(content)
            if self.sequences:
                names = re.compile(r"core\s+\d+:\s0x[0-9abcdef]+\s\(0x[0-9abcdef]+\)\s([\w.]+).*").findall(content)
        elif is_etiss:
            content = log_artifact.content
            if self.groups:
                encodings = re.compile(r"0x[0-9abcdef]+:\s\w+\s#\s([01]+)\s.*").findall(content)
                encodings = [f"0b{enc}" for enc in encodings]
            if self.sequences:
                names = re.compile(r"0x[0-9abcdef]+:\s(\w+)\s#\s[01]+\s.*").findall(content)
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
            lines = [f"{label};Count;Probablity"]
            for x in counts:
                line = f"{x};{counts[x]};{probs[x]:.3f}"
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
            ret_artifacts.append(artifact)
        if self.sequences:
            max_len = 3

            def _get_sublists(lst, length):
                ret = []
                for i in range(len(lst) - length + 1):
                    lst_ = lst[i : i + length]
                    ret.append(",".join(lst_))
                return ret

            for length in range(1, max_len + 1):
                names_ = _get_sublists(names, length)
                counts, probs = _helper(names_, top=self.top)
                sequence_csv = _gen_csv("Sequence", counts, probs)
                artifact = Artifact(
                    f"analyse_instructions_seq{length}.csv", content=sequence_csv, fmt=ArtifactFormat.TEXT
                )
                ret_artifacts.append(artifact)
        return ret_artifacts
