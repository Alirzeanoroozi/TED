Name="0_ted"

python dataset_export.py --dataset $Name
gsutil -m cp -r gs://export_pqt_$Name .
python dataset_indexer.py export_pqt_$Name/