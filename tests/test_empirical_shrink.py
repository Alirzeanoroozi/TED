import torch

from two_stage.diffusion import (
    deletion_segmentation_features,
    make_empirical_shrink_batch,
    pad_shrink_target_masks,
)


def _example_batch():
    sequences = ["ACDEFG"]
    assign = torch.tensor([[1, 0, 0, 2, 0, 2]])
    mask = torch.ones_like(assign, dtype=torch.bool)
    return sequences, assign, mask


def test_full_state_is_explicitly_sampled():
    sequences, assign, mask = _example_batch()
    examples = make_empirical_shrink_batch(
        sequences,
        assign,
        mask,
        full_state_probability=1.0,
        generator=torch.Generator().manual_seed(7),
    )
    example = examples[0]
    assert example.sequence == sequences[0]
    assert example.original_indices == list(range(6))
    assert example.target_mask == [False, True, True, False, True, False]
    assert example.already_deleted == 0
    assert example.remaining_budget == 3


def test_physical_shrink_never_deletes_domain_residues_and_keeps_origin_map():
    sequences, assign, mask = _example_batch()
    examples = make_empirical_shrink_batch(
        sequences,
        assign,
        mask,
        max_deleted_fraction=1.0,
        full_state_probability=0.0,
        generator=torch.Generator().manual_seed(11),
    )
    example = examples[0]
    assert {0, 3, 5}.issubset(set(example.original_indices))
    assert len(example.sequence) == 6 - example.already_deleted
    assert example.remaining_budget == sum(example.target_mask)
    assert example.already_deleted + example.remaining_budget == 3
    assert "".join(sequences[0][i] for i in example.original_indices) == example.sequence

    padded = pad_shrink_target_masks(examples, device=torch.device("cpu"))
    assert padded.shape == (1, len(example.sequence))
    assert int(padded.sum()) == example.remaining_budget


def test_deletion_features_are_masked_and_have_two_channels():
    logits = torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.0]])
    mask = torch.tensor([[True, True, True, True, False]])
    features = deletion_segmentation_features(logits, mask)
    assert features.shape == (1, 5, 2)
    assert torch.equal(features[0, 4], torch.zeros(2))
    assert features[0, 1, 1] > 0
    assert features[0, 2, 1] > 0

