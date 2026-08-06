from pathlib import Path
import sys


BENCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "chainsaw_cath1363_from_scratch"
)
sys.path.insert(0, str(BENCH_DIR))

from prepare_tedpred_benchmark import (  # noqa: E402
    reconstruct_authoritative_chopping,
    zero_based_to_ted_label,
)


def test_repeated_domain_names_form_one_discontinuous_domain():
    chopping, names = reconstruct_authoritative_chopping(
        "0-9|10-19|20-29", "1abcA_d1|1abcA_d2|1abcA_d1"
    )
    assert names == ["1abcA_d1", "1abcA_d2"]
    assert chopping == "0-9_20-29|10-19"
    assert zero_based_to_ted_label(chopping) == "1-10_21-30 | - * 11-20 | -"


def test_bounds_and_names_must_align():
    try:
        reconstruct_authoritative_chopping("0-9|10-19", "1abcA_d1")
    except ValueError as exc:
        assert "length mismatch" in str(exc)
    else:
        raise AssertionError("mismatched true_dbounds/true_dnames must fail")

