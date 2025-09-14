Name="0_ted"
Name_new="0_ted_new"

python dataset_export.py --dataset $Name
gsutil -m cp -r gs://export_pqt_$Name_new .
python dataset_indexer.py export_pqt_$Name_new/