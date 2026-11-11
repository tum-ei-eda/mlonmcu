from unittest.mock import Mock

import pytest

from mlonmcu.setup.tasks.spike import _validate_spikepk, _validate_spikepk_build


def test_validate_spikepk_skips_bare_metal_target():
    context = Mock()
    context.environment.has_target.side_effect = lambda name: name == "spike_bm"
    context.environment.vars = {}
    context.environment.repos = {}

    assert not _validate_spikepk(context)


def test_validate_spikepk_requires_proxy_kernel_repository():
    context = Mock()
    context.environment.has_target.side_effect = lambda name: name == "spike_pk"
    context.environment.vars = {}
    context.environment.repos = {}

    with pytest.raises(AssertionError, match="Undefined repository: 'spikepk'"):
        _validate_spikepk(context)


@pytest.mark.parametrize("xlen", [32, 64])
def test_validate_spikepk_build_skips_configured_proxy_kernel(xlen):
    context = Mock()
    context.environment.has_target.side_effect = lambda name: name == "spike_pk"
    context.environment.vars = {
        f"spike.pk_rv{xlen}": "/configured/pk",
        "spikepk.enable_rv32": xlen == 32,
        "spikepk.enable_rv64": xlen == 64,
    }

    assert not _validate_spikepk_build(context, params={"xlen": xlen})
