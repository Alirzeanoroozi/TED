import math

from benchmark.ted_eval import EvalConfig, evaluate_target


CFG = EvalConfig(input_indexing="one_based_inclusive", iou_threshold=0.8)


def test_iou_is_weighted_by_ground_truth_domain_size():
    # The prediction exactly recovers only the small 10-residue domain.
    result = evaluate_target(
        "1-90 | - * 91-100 | -",
        "91-100 | -",
        nres=100,
        config=CFG,
    )
    assert result["iou_chain"] == 0.1
    assert result["iou_chain_unweighted"] == 0.5
    assert result["correct_prop"] == 0.5


def test_overprediction_does_not_change_official_correct_denominator():
    result = evaluate_target(
        "1-50 | -",
        "1-50 | - * 51-100 | -",
        nres=100,
        config=CFG,
    )
    assert result["correct_prop"] == 1.0
    assert result["correct_prop_unweighted"] == 0.5


def test_unknown_cath_labels_are_not_counted_as_correct():
    result = evaluate_target("1-10 | -", "1-10 | -", nres=10, config=CFG)
    assert result["n_labeled_gt_domains"] == 0
    assert math.isnan(result["correct_cath_prop"])
    assert math.isnan(result["cath_level_score"])


def test_overlap_ranges_are_clipped_to_chain_length():
    result = evaluate_target("1-10 | -", "1-100 | -", nres=10, config=CFG)
    assert result["iou_chain"] == 1.0


def test_missing_prediction_is_a_failure_not_a_full_chain_domain():
    result = evaluate_target("1-10 | -", None, nres=10, config=CFG)
    assert result["pred_domain_count"] == 0
    assert result["iou_chain"] == 0.0

