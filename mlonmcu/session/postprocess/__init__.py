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
    AnalyseLinkerMapPostprocess,
    StageTimesGanttPostprocess,
    ProfileFunctionsPostprocess,
)

SUPPORTED_POSTPROCESSES = {}


def register_postprocess(postprocess_name, postprocess_cls=None, override=False):
    """Register a postprocess class under the given name."""

    def _register(cls):
        if postprocess_name in SUPPORTED_POSTPROCESSES and not override:
            raise RuntimeError(f"Postprocess {postprocess_name} is already registered")
        SUPPORTED_POSTPROCESSES[postprocess_name] = cls
        return cls

    if postprocess_cls is None:
        return _register
    return _register(postprocess_cls)


def get_postprocesses():
    """Return registered postprocesses."""
    return SUPPORTED_POSTPROCESSES


register_postprocess("filter_cols", FilterColumnsPostprocess)
register_postprocess("rename_cols", RenameColumnsPostprocess)
register_postprocess("features2cols", Features2ColumnsPostprocess)
register_postprocess("config2cols", Config2ColumnsPostprocess)
register_postprocess("mypost", MyPostprocess)
register_postprocess("passcfg2cols", PassConfig2ColumnsPostprocess)
register_postprocess("visualize", VisualizePostprocess)
register_postprocess("bytes2kb", Bytes2kBPostprocess)
register_postprocess("artifacts2cols", Artifact2ColumnPostprocess)
register_postprocess("analyse_instructions", AnalyseInstructionsPostprocess)
register_postprocess("compare_rows", CompareRowsPostprocess)
register_postprocess("analyse_dump", AnalyseDumpPostprocess)
register_postprocess("analyse_corev_counts", AnalyseCoreVCountsPostprocess)
register_postprocess("validate_outputs", ValidateOutputsPostprocess)
register_postprocess("validate_labels", ValidateLabelsPostprocess)
register_postprocess("export_outputs", ExportOutputsPostprocess)
register_postprocess("analyse_linker_map", AnalyseLinkerMapPostprocess)
register_postprocess("stage_times_gantt", StageTimesGanttPostprocess)
register_postprocess("profile_functions", ProfileFunctionsPostprocess)


__all__ = ["SUPPORTED_POSTPROCESSES", "register_postprocess", "get_postprocesses"]
