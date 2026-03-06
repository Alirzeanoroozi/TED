import os
import pandas as pd

parquet_files = [
    os.path.join("data/parquet_sequences", f) 
    for f in os.listdir("data/parquet_sequences")
    if f.endswith('.parquet')
]
df = pd.concat([pd.read_parquet(f) for f in parquet_files[:3]], ignore_index=True)
print(df.head())