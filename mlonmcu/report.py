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
"""Definitions of the Report class used by MLonMCU sessions and runs."""
from pathlib import Path
import pandas as pd


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 0)

SUPPORTED_FMTS = ["csv", "xlsx"]


class Report:
    """Report class wrapped around multiple pandas dataframes."""

    def __init__(self):
        self.pre_df = pd.DataFrame()
        self.main_df = pd.DataFrame()
        self.post_df = pd.DataFrame()

    @property
    def df(self):
        """Combine the three internal dataframes to a large one and return in."""
        # TODO: handle this properly by either adding NAN or use a single set(pre=, post=, main=) method
        return pd.concat([self.pre_df, self.main_df, self.post_df], axis=1)

    def export(self, path):
        """Export the report to  a file.

        Arguments
        ---------
        path : str
            Destination path.

        """
        ext = Path(path).suffix[1:]
        assert ext in SUPPORTED_FMTS, f"Unsupported report format: {ext}"
        parent = Path(path).parent
        if not parent.is_dir():
            parent.mkdir()
        if ext == "csv":
            self.df.to_csv(path, index=False)
        elif ext in ["xlsx", "xls"]:
            self.df.to_excel(path, index=False)
        else:
            raise RuntimeError()

    # def append(self, *args, **kwargs):
    #     self.df = self.df.append(*args, **kwargs, ignore_index=True)

    def set_pre(self, data):
        """Setter for the left third of the dataframe."""
        self.pre_df = pd.DataFrame.from_records(data).reset_index(drop=True)

    def set_post(self, data):
        """Setter for the right third of the dataframe."""
        self.post_df = pd.DataFrame.from_records(data).reset_index(drop=True)

    def set_main(self, data):
        """Setter for the center part of the dataframe."""
        self.main_df = pd.DataFrame.from_records(data).reset_index(drop=True)

    def set(self, pre=None, main=None, post=None):
        """Setter for the dataframe."""
        size = len(pre)
        self.set_pre(pre if pre is not None else {})
        if len(main) != size:
            assert len(main) == 0
            # self.set_main(pd.Series())
        else:
            self.set_main(main if main is not None else {})
        self.set_post(post if post is not None else {})

    def add(self, reports):
        """Helper function to append a line to an existing report."""
        if not isinstance(reports, list):
            reports = [reports]
        for report in reports:
            self.pre_df = pd.concat([self.pre_df, report.pre_df], axis=0).reset_index(drop=True)
            self.main_df = pd.concat([self.main_df, report.main_df], axis=0).reset_index(drop=True)
            self.post_df = pd.concat([self.post_df, report.post_df], axis=0).reset_index(drop=True)
