from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlonmcu.cli.common import kickoff_runs
from mlonmcu.session.run import RunStage


@pytest.mark.parametrize("executor", ["thread_pool", "process_pool", "cmdline", "context", "rpc"])
@pytest.mark.parametrize("override", [None, "0", "1"])
def test_cli_scheduling_default_and_explicit_override(executor, override):
    session = MagicMock()
    session.executor = executor
    session.runs = []
    context = SimpleNamespace(sessions=[session], environment=SimpleNamespace(vars={}))
    args = SimpleNamespace(
        config=[[f"runs_per_stage={override}"]] if override is not None else [],
        parallel=1,
        parallel_per_stage=None,
        progress=False,
        noop=False,
    )

    kickoff_runs(args, RunStage.RUN, context)

    expected = executor == "thread_pool" if override is None else override == "1"
    assert session.process_runs.call_args.kwargs["per_stage"] is expected
