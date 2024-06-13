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
"""MLonMCU postprocess submodule"""

from .postprocesses import (
    FilterColumnsPostprocess,
    RenameColumnsPostprocess,
    Features2ColumnsPostprocess,
    Config2ColumnsPostprocess,
    MyPostprocess,
    PassConfig2ColumnsPostprocess,
    VisualizePostprocess,
    Bytes2kBPostprocess,
    Artifact2ColumnPostprocess,
    AnalyseInstructionsPostprocess,
    CompareRowsPostprocess,
    AnalyseDumpPostprocess,
    AnalyseCoreVCountsPostprocess,
    ValidateOutputsPostprocess,
    ValidateLabelsPostprocess,
    ExportOutputsPostprocess,
)

SUPPORTED_POSTPROCESSES = {
    "filter_cols": FilterColumnsPostprocess,
    "rename_cols": RenameColumnsPostprocess,
    "features2cols": Features2ColumnsPostprocess,
    "config2cols": Config2ColumnsPostprocess,
    "mypost": MyPostprocess,
    "passcfg2cols": PassConfig2ColumnsPostprocess,
    "visualize": VisualizePostprocess,
    "bytes2kb": Bytes2kBPostprocess,
    "artifacts2cols": Artifact2ColumnPostprocess,
    "analyse_instructions": AnalyseInstructionsPostprocess,
    "compare_rows": CompareRowsPostprocess,
    "analyse_dump": AnalyseDumpPostprocess,
    "analyse_corev_counts": AnalyseCoreVCountsPostprocess,
    "validate_outputs": ValidateOutputsPostprocess,
    "validate_labels": ValidateLabelsPostprocess,
    "export_outputs": ExportOutputsPostprocess,
}
