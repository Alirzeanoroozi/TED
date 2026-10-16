import pandas as pd

df = pd.read_csv("results.csv")

df['pred_domains'] = df['prediction'].apply(lambda x: x.split('|')[0].strip())
df['pred_CATH'] = df['prediction'].apply(lambda x: x.split('|')[1].strip())
df['labe_domains'] = df['label'].apply(lambda x: x.split('|')[0].strip())
df['labe_CATH'] = df['label'].apply(lambda x: x.split('|')[1].strip())



