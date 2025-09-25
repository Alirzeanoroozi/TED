from parse_domains import parse_domains
from iou import chain_iou
import pandas as pd


def get_iou(predicted_chopped, actual_chopped):
    try:
        parsed_predicted_bounds, parsed_predicted_cath, parsed_actual_bounds, parsed_actual_cath = parse_domains(predicted_chopped, actual_chopped)
    except ValueError as e:
        return 0, 0, 0

    def to_set(domains_boundries):
        out_list = []
        for domains in domains_boundries:
            residue_indices = set()
            for boundry in domains:
                start_end = boundry.split("-")
                start = start_end[0]
                end = start_end[1]
                residue_indices.update(range(int(start), int(end) + 1))
            out_list.append(residue_indices)
        return out_list
    
    predicted = to_set(parsed_predicted_bounds)
    ground_truth = to_set(parsed_actual_bounds)
    
    correct_cath = 0
    for predicted_cath_item, actual_cath_item in zip(parsed_predicted_cath, parsed_actual_cath):
        if predicted_cath_item == actual_cath_item:
            correct_cath += 1
    correct_cath = correct_cath / len(parsed_predicted_cath)
    
    iou_chain, correct_prop = chain_iou(ground_truth, predicted)
    return iou_chain, correct_prop, correct_cath

if __name__ == "__main__":
    df = pd.read_csv("wandb_export_2025-09-23T10_04_42.561+03_00.csv")
    df['iou_chain'], df['correct_prop'], df['correct_cath'] = zip(*df.apply(lambda row: get_iou(row['predicted'], row['label']), axis=1))
    df.to_csv("results_iou.csv", index=False)
    
    print(df['iou_chain'].mean())
    print(df['correct_prop'].mean())
    print(df['correct_cath'].mean())
    
    iou_chain_nonzero = df[df['iou_chain'] != 0]
    correct_prop_nonzero = df[df['correct_prop'] != 0]
    correct_cath_nonzero = df[df['correct_cath'] != 0]

    print(iou_chain_nonzero['iou_chain'].mean() if not iou_chain_nonzero.empty else 0)
    print(correct_prop_nonzero['correct_prop'].mean() if not correct_prop_nonzero.empty else 0)
    print(correct_cath_nonzero['correct_cath'].mean() if not correct_cath_nonzero.empty else 0)