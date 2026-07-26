import pytest
import yaml

from genmol.rl.trainer import load_config


def _write_config(tmp_path, **overrides):
    config = {
        'init_ckpt_path': 'unused.ckpt',
        'candidate_diversity_reward_weight': 0.9,
    }
    config.update(overrides)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_candidate_diversity_config_accepts_raw_loo_weight(tmp_path):
    config = load_config(_write_config(tmp_path))
    assert config.rl_algorithm == 'coupled_grpo'
    assert config.candidate_diversity_reward_weight == pytest.approx(0.9)


@pytest.mark.parametrize('weight', [-0.1, 1.1])
def test_candidate_diversity_config_rejects_out_of_range_weight(
    tmp_path,
    weight,
):
    with pytest.raises(
        ValueError,
        match='candidate_diversity_reward_weight must be in',
    ):
        load_config(
            _write_config(
                tmp_path,
                candidate_diversity_reward_weight=weight,
            )
        )


@pytest.mark.parametrize(
    'overrides, expected_option',
    [
        ({'hbd': True}, 'hbd'),
        ({'diverse_minibatch': True}, 'diverse_minibatch'),
        (
            {'entropy_regularization_weight': 0.01},
            'entropy_regularization_weight',
        ),
        (
            {'diversity_regularizer_weight': 0.1},
            'diversity_regularizer_weight',
        ),
    ],
)
def test_candidate_diversity_config_rejects_variant_stacking(
    tmp_path,
    overrides,
    expected_option,
):
    with pytest.raises(ValueError, match=expected_option):
        load_config(_write_config(tmp_path, **overrides))


def test_candidate_diversity_config_rejects_sgrpo(tmp_path):
    with pytest.raises(
        ValueError,
        match='only supported when rl_algorithm=coupled_grpo',
    ):
        load_config(
            _write_config(
                tmp_path,
                rl_algorithm='coupled_sgrpo',
            )
        )
