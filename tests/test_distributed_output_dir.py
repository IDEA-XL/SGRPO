from types import SimpleNamespace

from genmol.mm.trainer import resolve_output_dir as resolve_mm_output_dir
from genmol.rl.trainer import resolve_output_dir as resolve_denovo_output_dir


def test_denovo_output_dir_is_shared_by_slurm_ranks(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '12345')

    output_dir = resolve_denovo_output_dir(
        SimpleNamespace(output_dir=None),
        '/tmp/denovo.yaml',
    )

    assert output_dir.endswith('/denovo_slurm12345')


def test_mm_output_dir_is_shared_by_slurm_ranks(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '67890')

    output_dir = resolve_mm_output_dir(
        SimpleNamespace(output_dir=None),
        '/tmp/mmgenmol.yaml',
    )

    assert output_dir.endswith('/mmgenmol_slurm67890')


def test_explicit_output_dir_takes_precedence_over_slurm_job(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    config = SimpleNamespace(output_dir='/tmp/explicit-run')

    assert resolve_denovo_output_dir(config, '/tmp/denovo.yaml') == '/tmp/explicit-run'
    assert resolve_mm_output_dir(config, '/tmp/mmgenmol.yaml') == '/tmp/explicit-run'
