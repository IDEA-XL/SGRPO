import json
import pickle

import pytest

from genmol.rl.motif import (
    MotifRecord,
    canonicalize_single_attachment_fragment,
    load_motif_records,
    molecule_contains_fragment,
    sample_group_motif_records,
)
from genmol.rl.specs import sample_conditioned_group_specs


def test_canonicalize_single_attachment_fragment_normalizes_dummy_label():
    canonical = canonicalize_single_attachment_fragment("[*:1]CCO")
    assert canonical.count("*") == 1
    assert "[1*]" in canonical


def test_canonicalize_single_attachment_fragment_rejects_multiple_dummies():
    with pytest.raises(ValueError, match="exactly one"):
        canonicalize_single_attachment_fragment("[*:1]CC[*:2]")


def test_molecule_contains_fragment_ignores_attachment_dummy():
    assert molecule_contains_fragment("CCOCC", "[*:1]CCO")
    assert not molecule_contains_fragment("c1ccccc1", "[*:1]CCO")


def test_molecule_contains_aromatic_n_attachment_fragment():
    fragment = "[1*]n1[nH]c2ccccc2c1=O"
    assert molecule_contains_fragment(
        "Cn1[nH]c2ccccc2c1=O",
        fragment,
    )
    assert not molecule_contains_fragment("c1ccccc1", fragment)


def test_load_motif_records_is_strict_and_canonical(tmp_path):
    path = tmp_path / "motifs.jsonl"
    path.write_text(
        json.dumps(
            {
                "motif_id": "motif-1",
                "smiles": "[*:1]CCO",
                "frequency": 7,
            }
        )
        + "\n"
    )
    records = load_motif_records(path)
    assert records == (
        MotifRecord(
            motif_id="motif-1",
            smiles=canonicalize_single_attachment_fragment("[*:1]CCO"),
            frequency=7,
        ),
    )


def test_conditioned_spec_accounts_for_prompt_length(tmp_path):
    length_path = tmp_path / "lengths.pk"
    with length_path.open("wb") as handle:
        pickle.dump([30], handle)

    specs = sample_conditioned_group_specs(
        base_sequence_lengths=[7, 17],
        generation_temperature=1.0,
        randomness=0.3,
        min_add_len=5,
        seed=123,
        length_path=str(length_path),
    )

    assert [spec.add_seq_len for spec in specs] == [23, 13]


def test_motif_sampling_shares_condition_within_supergroup():
    records = (
        MotifRecord("a", "[1*]CC"),
        MotifRecord("b", "[1*]CO"),
    )
    sampled = sample_group_motif_records(
        records,
        num_groups=8,
        supergroup_num_groups=4,
        seed=123,
    )
    assert len(sampled) == 8
    assert sampled[:4] == [sampled[0]] * 4
    assert sampled[4:] == [sampled[4]] * 4
