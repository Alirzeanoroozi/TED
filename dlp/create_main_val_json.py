import pandas as pd
import json

df = pd.read_csv("../results/results.csv")


with open("jsons/main_val.json", "a") as f:
    for index, row in df.iterrows():
        output_dict = {
            "Sequence": row["input"],
            "label": row["label"]
        }
        json.dump(output_dict, f)
        f.write("\n")
