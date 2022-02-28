"""MLonMCU postprocess submodule"""

from .postprocesses import AverageCyclesPostprocess, DetailedCyclesPostprocess, FilterColumnsPostprocess

SUPPORTED_POSTPROCESSES = {
    "average_cycles": AverageCyclesPostprocess,
    "detailed_cycles": DetailedCyclesPostprocess,
    "filter_cols": FilterColumnsPostprocess,
}
