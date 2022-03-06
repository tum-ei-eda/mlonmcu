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
import pandas as pd
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from mlonmcu.artifact import Artifact, ArtifactFormat
import ast

from .postprocess import SessionPostprocess, RunPostprocess
from mlonmcu.logging import get_logger

logger = get_logger()


def match_rows(df, cols):
    groups = df.astype(str).groupby(cols).apply(lambda x: tuple(x.index)).tolist()
    return groups


class AverageCyclesPostprocess(SessionPostprocess, RunPostprocess):

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "merge_rows": False,
    }

    def __init__(self, features=None, config=None):
        super().__init__("average_cycles", features=features, config=config)

    @property
    def merge_rows(self):
        return bool(self.config["merge_rows"])

    def post_run(self, report):
        if not self.merge_rows:
            if "Total Cycles" not in report.main_df or "Num" not in report.pre_df:
                return
            report.main_df["Average Cycles"] = report.main_df["Total Cycles"] / report.pre_df["Num"]

    def post_session(self, report):
        if self.merge_rows:
            if "Total Cycles" not in report.main_df or "Num" not in report.pre_df:
                return
            ignore_cols = ["Session", "Run", "Num", "Comment"]
            combined_df = pd.concat([report.pre_df, report.post_df], axis=1)
            use_cols = combined_df.columns
            use_cols = list(filter(lambda elem: elem not in ignore_cols, use_cols))
            groups = match_rows(combined_df, use_cols)
            to_drop = []
            for group in groups:
                total_num = report.pre_df["Num"][list(group)].sum()
                total_cycles = report.main_df["Total Cycles"][list(group)].sum()
                avg_cycles = total_cycles / total_num
                report.pre_df["Num"][[group[0]]] = total_num
                report.main_df.loc[group[0], "Average Cycles"] = avg_cycles
                to_drop.extend(list(group[1:]))
            report.pre_df.drop(to_drop, inplace=True)
            report.main_df.drop(to_drop, inplace=True)
            report.post_df.drop(to_drop, inplace=True)


class DetailedCyclesPostprocess(SessionPostprocess):

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "warn": True,
    }

    def __init__(self, features=None, config=None):
        super().__init__("detailed_cycles", features=features, config=config)

    @property
    def warn(self):
        return bool(self.config["warn"])

    def get_detailed_cycles(self, low_num, low_cycles, high_num, high_cycles):
        assert high_cycles > low_cycles
        assert high_num > low_num
        diff_cycles = high_cycles - low_cycles
        diff_num = high_num - low_num
        invoke_cycles = int(float(diff_cycles) / (diff_num))
        setup_cycles = low_cycles - (invoke_cycles * low_num)
        return setup_cycles, invoke_cycles

    def post_session(self, report):
        if "Total Cycles" not in report.main_df or "Num" not in report.pre_df:
            if self.warn:
                logger.warning(f"Postprocess '{self.name}' was not applied because of missing columns")
            return
        ignore_cols = ["Session", "Run", "Num", "Comment"]
        combined_df = pd.concat([report.pre_df, report.post_df], axis=1)
        use_cols = combined_df.columns
        use_cols = list(filter(lambda elem: elem not in ignore_cols, use_cols))
        groups = match_rows(combined_df, use_cols)
        to_drop = []
        for group in groups:
            if len(group) == 1:
                continue
                if self.warn:
                    logger.warning("Unable to find a suitable pair for extracting detailed cycle counts")
            max_idx = report.pre_df["Num"][list(group)].idxmax()
            min_idx = report.pre_df["Num"][list(group)].idxmin()
            assert max_idx != min_idx
            max_cycles = report.main_df["Total Cycles"][max_idx]
            min_cycles = report.main_df["Total Cycles"][min_idx]
            max_num = report.pre_df["Num"][max_idx]
            min_num = report.pre_df["Num"][min_idx]
            setup_cycles, invoke_cycles = self.get_detailed_cycles(min_num, min_cycles, max_num, max_cycles)
            report.main_df.loc[max_idx, "Setup Cycles"] = setup_cycles
            report.main_df.loc[max_idx, "Invoke Cycles"] = invoke_cycles
            to_drop.extend([idx for idx in group if idx != max_idx])
        report.pre_df.drop(to_drop, inplace=True)
        report.main_df.drop(to_drop, inplace=True)
        report.post_df.drop(to_drop, inplace=True)


class FilterColumnsPostprocess(SessionPostprocess):

    DEFAULTS = {
        **SessionPostprocess.DEFAULTS,
        "keep": None,
        "drop": None,
        "drop_nan": False,
        "drop_const": False,
    }

    def __init__(self, features=None, config=None):
        super().__init__("filter_cols", features=features, config=config)

    @property
    def keep(self):
        cfg = self.config["keep"]
        if isinstance(cfg, str):
            return ast.literal_eval(cfg)
        return cfg

    @property
    def drop(self):
        cfg = self.config["drop"]
        if isinstance(cfg, str):
            return ast.literal_eval(cfg)
        return cfg

    @property
    def drop_nan(self):
        return bool(self.config["drop_nan"])

    @property
    def drop_const(self):
        return bool(self.config["drop_const"])

    def post_session(self, report):
        def _filter_df(df, keep, drop, drop_nan=False, drop_const=False):
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
            report.pre_df, self.keep, self.drop, drop_nan=self.drop_nan, drop_const=self.drop_const
        )
        report.main_df = _filter_df(
            report.main_df, self.keep, self.drop, drop_nan=self.drop_nan, drop_const=self.drop_const
        )
        report.post_df = _filter_df(
            report.post_df, self.keep, self.drop, drop_nan=self.drop_nan, drop_const=self.drop_const
        )


class Features2ColumnsPostprocess(SessionPostprocess):  # RunPostprocess?
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
    def __init__(self, features=None, config=None):
        super().__init__("config2cols", features=features, config=config)

    def post_session(self, report):
        df = report.post_df
        if "Config" not in df.columns:
            return
        config_df = df["Config"].apply(pd.Series).add_prefix("config_")
        tmp_df = df.drop(columns=["Config"])
        new_df = pd.concat([tmp_df, config_df], axis=1)
        report.post_df = new_df


class Bytes2kBPostprocess(SessionPostprocess):  # RunPostprocess?
    def __init__(self, features=None, config=None):
        super().__init__("bytes2kb", features=features, config=config)

    def post_session(self, report):
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
        return self.config["format"]

    def post_session(self, report):
        df = pd.concat([report.pre_df, report.main_df], axis=1)

        if self.format != "png":
            raise NotImplementedError("Currently only supports PNG")

        import matplotlib.pyplot as plt

        COLS = ["Total Cycles", "Total ROM", "Total RAM"]
        for col in COLS:
            if col not in report.main_df.columns:
                return
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
