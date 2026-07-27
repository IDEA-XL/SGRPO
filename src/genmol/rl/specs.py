import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from functools import lru_cache

from rl_shared.sampling import sample_scalar_or_range


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


@dataclass(frozen=True)
class DeNovoSpec:
    add_seq_len: int
    generation_temperature: float
    randomness: float
    min_add_len: int


@lru_cache(maxsize=1)
def load_length_distribution(path=None):
    length_path = path or os.path.join(ROOT_DIR, 'data', 'len.pk')
    with open(length_path, 'rb') as handle:
        return tuple(pickle.load(handle))


def sample_add_seq_len(min_add_len, rng, max_completion_length=None, length_path=None):
    seq_len_list = load_length_distribution(length_path)
    sampled_total_length = rng.choice(seq_len_list)
    add_seq_len = max(sampled_total_length - 2, min_add_len)
    if max_completion_length is not None:
        add_seq_len = min(add_seq_len, max_completion_length)
    return add_seq_len


def sample_conditioned_add_seq_len(
    base_sequence_length,
    min_add_len,
    rng,
    max_completion_length=None,
    length_path=None,
):
    if base_sequence_length < 2:
        raise ValueError('base_sequence_length must include at least BOS and EOS')
    seq_len_list = load_length_distribution(length_path)
    sampled_total_length = rng.choice(seq_len_list)
    add_seq_len = max(sampled_total_length - base_sequence_length, min_add_len)
    if max_completion_length is not None:
        add_seq_len = min(add_seq_len, max_completion_length)
    return add_seq_len


def sample_conditioned_group_specs(
    base_sequence_lengths,
    generation_temperature,
    randomness,
    min_add_len,
    seed,
    max_completion_length=None,
    length_path=None,
):
    if not base_sequence_lengths:
        raise ValueError('base_sequence_lengths must be non-empty')
    rng = random.Random(seed)
    specs = []
    for base_sequence_length in base_sequence_lengths:
        specs.append(
            DeNovoSpec(
                add_seq_len=sample_conditioned_add_seq_len(
                    base_sequence_length=base_sequence_length,
                    min_add_len=min_add_len,
                    rng=rng,
                    max_completion_length=max_completion_length,
                    length_path=length_path,
                ),
                generation_temperature=sample_scalar_or_range(
                    generation_temperature,
                    rng,
                    name='generation_temperature',
                    min_exclusive=0.0,
                ),
                randomness=sample_scalar_or_range(
                    randomness,
                    rng,
                    name='randomness',
                    min_exclusive=0.0,
                ),
                min_add_len=min_add_len,
            )
        )
    return specs


def sample_group_specs(
    num_groups,
    generation_temperature,
    randomness,
    min_add_len,
    seed,
    max_completion_length=None,
    length_path=None,
):
    if num_groups <= 0:
        raise ValueError('num_groups must be positive')

    rng = random.Random(seed)
    specs = []
    for _ in range(num_groups):
        specs.append(
            DeNovoSpec(
                add_seq_len=sample_add_seq_len(
                    min_add_len=min_add_len,
                    rng=rng,
                    max_completion_length=max_completion_length,
                    length_path=length_path,
                ),
                generation_temperature=sample_scalar_or_range(
                    generation_temperature,
                    rng,
                    name='generation_temperature',
                    min_exclusive=0.0,
                ),
                randomness=sample_scalar_or_range(
                    randomness,
                    rng,
                    name='randomness',
                    min_exclusive=0.0,
                ),
                min_add_len=min_add_len,
            )
        )
    return specs


def sample_supergroup_shared_specs(
    num_groups,
    supergroup_num_groups,
    generation_temperature,
    randomness,
    min_add_len,
    seed,
    max_completion_length=None,
    length_path=None,
):
    if supergroup_num_groups <= 1:
        raise ValueError('supergroup_num_groups must be greater than 1')
    if num_groups <= 0:
        raise ValueError('num_groups must be positive')
    if num_groups % supergroup_num_groups != 0:
        raise ValueError(
            'num_groups must be divisible by supergroup_num_groups: '
            f'{num_groups} vs {supergroup_num_groups}'
        )

    num_supergroups = num_groups // supergroup_num_groups
    base_specs = sample_group_specs(
        num_groups=num_supergroups,
        generation_temperature=generation_temperature,
        randomness=randomness,
        min_add_len=min_add_len,
        seed=seed,
        max_completion_length=max_completion_length,
        length_path=length_path,
    )
    specs = []
    for spec in base_specs:
        specs.extend([spec] * supergroup_num_groups)
    return specs


def expand_group_specs(group_specs, num_generations):
    if num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')

    expanded = []
    for spec in group_specs:
        expanded.extend([spec] * num_generations)
    return expanded


def serialize_specs(specs):
    return json.dumps([asdict(spec) for spec in specs], sort_keys=True)


def deserialize_specs(payload):
    return [DeNovoSpec(**item) for item in json.loads(payload)]
