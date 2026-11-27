"""Incomplete validation must not yield a successful agreement fraction."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.session.postprocess.postprocesses import ValidateOutputsPostprocess


@pytest.mark.parametrize("actual,reference", [(0, 0), (1, 2), (2, 1)])
def test_validate_outputs_rejects_missing_or_extra_samples(tmp_path, actual, reference):
    artifacts = [Artifact("model_info.yml", content=yaml.safe_dump({"output_names": ["y"]}), fmt=ArtifactFormat.TEXT)]
    for name, count in [("outputs.npy", actual), ("outputs_ref.npy", reference)]:
        path = tmp_path / name
        np.save(path, [{"y": np.zeros([1], dtype="float32")} for _ in range(count)])
        artifacts.append(Artifact(name, path=path, raw=path.read_bytes(), fmt=ArtifactFormat.BIN))
    report = SimpleNamespace(post_df=pd.DataFrame([{}]))
    with pytest.raises(ValueError, match="sample count mismatch"):
        ValidateOutputsPostprocess().post_run(report, artifacts)
