import numpy as np
from itertools import product
from scipy.optimize import linear_sum_assignment

def iou(set_a, set_b):
    """Compute IoU between two sets."""
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

def chain_iou(ground_truth_domains, predicted_domains):
    """
    Compute IoU_chain and correctly parsed proportion.

    Parameters
    ----------
    ground_truth_domains : list of sets
        Each set contains residue indices for a ground-truth domain.
    predicted_domains : list of sets
        Each set contains residue indices for a predicted domain.

    Returns
    -------
    iou_chain : float
        Weighted chain-level IoU.
    correct_prop : float
        Proportion of ground-truth domains with IoU >= 0.8.
    """
    n_gt = len(ground_truth_domains)
    n_pred = len(predicted_domains)

    # Build IoU matrix
    iou_matrix = np.zeros((n_gt, n_pred))
    for i, gt in enumerate(ground_truth_domains):
        for j, pred in enumerate(predicted_domains):
            iou_matrix[i, j] = iou(gt, pred)

    # We want to maximize IoU sum, so we negate for Hungarian algo (which minimizes cost)
    cost_matrix = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Paired IoUs
    paired_ious = [iou_matrix[i, j] for i, j in zip(row_ind, col_ind) if i < n_gt and j < n_pred]

    # Compute weighted IoU_chain
    domain_sizes = [len(gt) for gt in ground_truth_domains]
    total_residues = sum(domain_sizes)
    weights = np.array(domain_sizes) / total_residues

    iou_chain = sum(p * w for p, w in zip(paired_ious, weights))

    # Correctly parsed domains (IoU >= 0.8)
    correct_count = sum(1 for p in paired_ious if p >= 0.8)
    correct_prop = correct_count / n_gt if n_gt > 0 else 0.0

    return iou_chain, correct_prop

if __name__ == "__main__":
    # Example usage:
    from parse_domains import parse_domains
    parsed_predicted, parsed_actual = parse_domains("55-128_165-214_251-341 | 2.40.128.100" ,"2-111_193-240 | 2.40.160.20")


    def to_set(boundries):
        out_list = []
        print(boundries)
        for boundry in boundries:
            start_end = boundry.split("-")
            start = start_end[0]
            end = start_end[1]
            out_list.append(set(range(int(start), int(end) + 1)))
        return out_list

    print(parsed_predicted, parsed_actual)
    predicted_boundries = parsed_predicted[0][0]
    actual_boundries = parsed_actual[0][0]
    predicted = to_set(predicted_boundries)
    ground_truth = to_set(actual_boundries)

    print(predicted, ground_truth)
    # ground_truth = [{1,2,3,4,5}, {10,11,12,13}]
    # predicted    = [{2,3,4,5,6}, {10,11,12}]

    iou_chain, correct_prop = chain_iou(ground_truth, predicted)
    print("Chain IoU:", iou_chain)
    print("Correctly parsed proportion:", correct_prop)