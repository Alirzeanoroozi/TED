import pandas as pd

from two_stage.experiment import build_split_manifest, infer_split_groups


def test_augmented_rows_share_the_original_protein_group():
    df = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P1__aug_01", "P1__aug_02", "P2"],
            "augmented_from": [None, "P1", "P1", None],
            "sequence": ["AAAA", "AAA", "AA", "CCCC"],
            "chopping_star": ["1-4 | -", "1-3 | -", "1-2 | -", "1-4 | -"],
        }
    )
    groups, source = infer_split_groups(df)
    assert source == "augmentation_source"
    assert groups.iloc[:3].nunique() == 1
    assert groups.iloc[0] == "P1"


def test_homology_cluster_is_preferred_and_never_crosses_splits():
    rows = []
    for cluster in range(40):
        for member in range(2):
            rows.append(
                {
                    "homology_cluster": f"c{cluster}",
                    "uniprot_id": f"P{cluster}_{member}",
                    "sequence": "A" * (10 + member),
                    "chopping_star": f"1-{10 + member} | -",
                }
            )
    df = pd.DataFrame(rows)
    manifest = build_split_manifest(
        df, val_ratio=0.2, test_ratio=0.2, seed=42, group_col="auto"
    )
    assert set(manifest["split_group_source"]) == {"homology_cluster"}
    assert manifest.groupby("split_group")["split"].nunique().max() == 1
    assert set(manifest["split"]) == {"train", "val", "test"}
