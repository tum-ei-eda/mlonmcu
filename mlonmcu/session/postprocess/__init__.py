"""MLonMCU postprocess submodule"""

from .postprocesses import (
    AverageCyclesPostprocess,
    DetailedCyclesPostprocess,
    FilterColumnsPostprocess,
    Features2ColumnsPostprocess,
    Config2ColumnsPostprocess,
    VisualizePostprocess,
    Bytes2kBPostprocess,
)

SUPPORTED_POSTPROCESSES = {
    "average_cycles": AverageCyclesPostprocess,
    "detailed_cycles": DetailedCyclesPostprocess,
    "filter_cols": FilterColumnsPostprocess,
    "features2cols": Features2ColumnsPostprocess,
    "config2cols": Config2ColumnsPostprocess,
    "visualize": VisualizePostprocess,
    "bytes2kb": Bytes2kBPostprocess,
}
