from pathlib import Path
import json
import pytest
from mlonmcu.feature.features import get_available_features
from mlonmcu.target.riscv.spike import SpikeBMTarget, SpikePKTarget
from mlonmcu.target.riscv import spike


@pytest.mark.parametrize('cls', [SpikeBMTarget, SpikePKTarget])
@pytest.mark.parametrize('xlen', [32, 64])
def test_plugin_arguments_and_logging_preserve_existing_args(tmp_path, monkeypatch, cls, xlen):
    name = 'spike_bm' if cls is SpikeBMTarget else 'spike_pk'
    library = tmp_path / 'extension with spaces.so'
    library.touch()
    pk = tmp_path / 'pk'
    pk.touch()
    feature = get_available_features()['log_instrs'](config={'log_instrs.to_file': True})
    target = cls(features=[feature], config={
        f'{name}.xlen': xlen, f'{name}.arch': f'rv{xlen}im_xisaac',
        f'{name}.extlibs': [str(library)], f'{name}.extension_names': ['m2_Custom'],
        f'{name}.extra_args': ['--log-commits'], 'spike.exe': '/bin/true', 'spike.pk': str(pk),
    })
    captured = []
    monkeypatch.setattr(spike, 'execute', lambda *args, **kwargs: captured.extend(args) or '')
    target.exec(tmp_path / 'program.elf', cwd=tmp_path)
    assert f'--extlib={library}' in captured
    assert '--extension=m2_Custom' in captured
    assert '--log-commits' in captured
    assert '-l' in captured
    assert '--log=instrs.txt' in captured
    assert next(arg for arg in captured if str(arg).startswith('--isa=')).find('xisaac') == -1
    assert (pk.resolve() in captured) == (cls is SpikePKTarget)


def test_missing_library_fails_before_simulator_launch(tmp_path):
    target = SpikeBMTarget(config={'spike_bm.extlibs': [str(tmp_path / 'missing.so')]})
    with pytest.raises(FileNotFoundError, match='extension library'):
        target.get_spike_args()


def test_plugin_manifest_resolves_library_relative_to_manifest(tmp_path):
    library = tmp_path / 'service-output' / 'libextension.so'
    library.parent.mkdir()
    library.touch()
    manifest = tmp_path / 'service-output' / 'spike_plugin.json'
    manifest.write_text(json.dumps({
        'schema': 1, 'core': 'XIsaacCore', 'extlibs': ['libextension.so'],
        'extension_names': ['m2_XIsaacCore'],
    }))
    target = SpikeBMTarget(config={'spike_bm.plugin_manifest': str(manifest)})
    args = target.get_spike_args()
    assert f'--extlib={library.resolve()}' in args
