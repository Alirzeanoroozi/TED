"""Sequence-only inference and evaluation for residue MLM."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from benchmark.ted_eval import EvalConfig, evaluate_target
from residue_mlm.decode import batch_predictions_to_chopping_stars, infer_num_domains
from residue_mlm.labels import MASK_ID, LevelVocabulary, PAD_ID
from residue_mlm.model import ResidueMLM


def build_sequence_only_inputs(
    batch: Dict[str, torch.Tensor],
    *,
    max_len: int,
) -> Dict[str, torch.Tensor]:
    device = batch["aa_ids"].device
    batch_size = batch["aa_ids"].size(0)
    attention_mask = torch.arange(max_len, device=device).unsqueeze(0) < batch["seq_len"].unsqueeze(1)

    domain_ids_input = torch.zeros_like(batch["domain_ids"])
    cath_inputs = {
        level: torch.full_like(batch[level], MASK_ID)
        for level in ("cath_c", "cath_a", "cath_t", "cath_h")
    }

    return {
        "aa_ids_input": batch["aa_ids"],
        "positions": batch["positions"],
        "num_domains": torch.zeros(batch_size, dtype=torch.long, device=device),
        "domain_ids_input": domain_ids_input,
        "cath_inputs": cath_inputs,
        "attention_mask": attention_mask,
    }


def refine_sequence_inputs_with_predicted_domain_count(
    model_inputs: Dict[str, torch.Tensor],
    pred_domain_ids: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    refined = dict(model_inputs)
    num_domains = torch.zeros_like(model_inputs["num_domains"])
    for batch_idx in range(pred_domain_ids.size(0)):
        num_domains[batch_idx] = infer_num_domains(
            pred_domain_ids[batch_idx],
            int(seq_lens[batch_idx]),
        )
    refined["num_domains"] = num_domains
    return refined


@torch.no_grad()
def predict_residue_labels(
    model: ResidueMLM,
    batch: Dict[str, torch.Tensor],
    *,
    refine_domain_count: bool = True,
) -> Dict[str, torch.Tensor]:
    model_inputs = build_sequence_only_inputs(batch, max_len=batch["positions"].size(1))
    logits = model(**model_inputs)
    pred_domain = logits["domain_logits"].argmax(dim=-1)

    if refine_domain_count:
        model_inputs = refine_sequence_inputs_with_predicted_domain_count(
            model_inputs,
            pred_domain,
            batch["seq_len"],
        )
        logits = model(**model_inputs)
        pred_domain = logits["domain_logits"].argmax(dim=-1)

    pred_cath = {
        level: logits[f"{level}_logits"].argmax(dim=-1)
        for level in ("cath_c", "cath_a", "cath_t", "cath_h")
    }
    return {
        "domain_ids": pred_domain,
        **pred_cath,
    }


def compute_token_accuracies(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    seq_lens: torch.Tensor,
    *,
    mlm_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Compute masked-token accuracy for each prediction head."""
    max_len = seq_lens.max().item()
    device = seq_lens.device
    valid = torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)

    metrics: Dict[str, float] = {}
    task_map = {
        "aa": ("aa_logits", "aa_labels"),
        "domain": ("domain_logits", "domain_labels"),
        "cath_c": ("cath_c_logits", "cath_c_labels"),
        "cath_a": ("cath_a_logits", "cath_a_labels"),
        "cath_t": ("cath_t_logits", "cath_t_labels"),
        "cath_h": ("cath_h_logits", "cath_h_labels"),
    }

    for task_name, (logit_key, label_key) in task_map.items():
        if label_key not in labels:
            continue
        label = labels[label_key]
        pred = logits[logit_key].argmax(dim=-1)
        supervised = valid & (label != -100)
        if mlm_mask is not None and task_name != "domain":
            supervised = supervised & mlm_mask
        if task_name == "domain":
            supervised = supervised & (label != -100)

        count = int(supervised.sum().item())
        if count == 0:
            metrics[f"{task_name}_acc"] = 0.0
            continue
        correct = int((pred[supervised] == label[supervised]).sum().item())
        metrics[f"{task_name}_acc"] = correct / count

    if all(key in metrics for key in ("cath_c_acc", "cath_a_acc", "cath_t_acc", "cath_h_acc")):
        metrics["cath_full_acc"] = (
            metrics["cath_c_acc"]
            + metrics["cath_a_acc"]
            + metrics["cath_t_acc"]
            + metrics["cath_h_acc"]
        ) / 4.0
    return metrics


def compute_residue_boundary_accuracy(
    pred_domain_ids: torch.Tensor,
    gt_domain_ids: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Dict[str, float]:
    """Per-residue domain-id match and boundary (change-point) match."""
    device = pred_domain_ids.device
    max_len = pred_domain_ids.size(1)
    valid = torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)

    residue_match = (pred_domain_ids == gt_domain_ids) & valid
    residue_total = int(valid.sum().item())
    residue_acc = float(residue_match.sum().item()) / residue_total if residue_total else 0.0

    boundary_match_count = 0
    boundary_total = 0
    for batch_idx in range(pred_domain_ids.size(0)):
        length = int(seq_lens[batch_idx])
        if length < 2:
            continue
        gt = gt_domain_ids[batch_idx, :length]
        pred = pred_domain_ids[batch_idx, :length]
        gt_boundary = gt[1:] != gt[:-1]
        pred_boundary = pred[1:] != pred[:-1]
        boundary_match_count += int((gt_boundary == pred_boundary).sum().item())
        boundary_total += length - 1

    boundary_acc = boundary_match_count / boundary_total if boundary_total else 0.0
    return {
        "residue_domain_acc": residue_acc,
        "boundary_acc": boundary_acc,
    }


def evaluate_sequence_predictions(
    model: ResidueMLM,
    batch: Dict[str, torch.Tensor],
    cath_vocabs: Dict[str, LevelVocabulary],
    chopping_stars: List[str],
    left_offsets: List[int],
    *,
    eval_config: Optional[EvalConfig] = None,
) -> Dict[str, float]:
    """Predict chopping_star from sequence and compare to ground truth."""
    cfg = eval_config or EvalConfig(input_indexing="one_based_inclusive")
    predictions = predict_residue_labels(model, batch)
    pred_chopping_stars = batch_predictions_to_chopping_stars(
        predictions["domain_ids"],
        predictions,
        batch["seq_len"],
        left_offsets,
        cath_vocabs,
    )

    chain_metrics = {
        "iou_chain": [],
        "correct_cath": [],
        "cath_level_score": [],
        "boundary_distance_score": [],
        "domain_count_match": [],
        "pred_parse_ok": [],
    }

    for pred, gt, seq_len in zip(pred_chopping_stars, chopping_stars, batch["seq_len"].tolist()):
        metrics = evaluate_target(
            gt,
            pred,
            nres=int(seq_len),
            config=cfg,
        )
        for key in chain_metrics:
            value = metrics.get(key, 0.0)
            if key == "pred_parse_ok":
                chain_metrics[key].append(float(bool(value)))
            else:
                chain_metrics[key].append(float(value))

    return {key: sum(values) / len(values) if values else 0.0 for key, values in chain_metrics.items()}
