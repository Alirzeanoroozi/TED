import os
from google.cloud import bigquery, storage
from google.cloud.storage import Bucket
import sys

GCP_KEYFILE_PATH = 'key.json'
PROJECT_ID = "pfsdb3"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEYFILE_PATH

bq_client = bigquery.Client()

def make_export_url(table: str, bucket: str):
    return f"gs://{bucket}/{table}/*"

def list_tables(dataset):
    dataset_id = f'{PROJECT_ID}.{dataset}'
    tables = bq_client.list_tables(dataset_id)

    # Convert iterator to list first to avoid consuming it twice
    table_list = list(tables)
    
    print("Tables contained in '{}':".format(dataset_id))
    for table in table_list:
        print("\t{}".format(table.table_id))
    return [table.table_id for table in table_list]

def export_table(dataset, table, bucket):
    destination_uri = make_export_url(table, bucket)
    print(f'Bucket in progress: {destination_uri}')

    job_config = bigquery.ExtractJobConfig(destination_format="PARQUET", compression="GZIP")

    job = bq_client.extract_table(
        source=bq_client.dataset(dataset).table(table),
        destination_uris=[destination_uri],
        job_config=job_config,
    )

    job.result()

    print(f"Exported table {table} to {destination_uri}")

def create_bucket_class_location(bucket_name):
    """
    Create a new bucket in the US region with the coldline storage
    class
    """

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = "Standard"
    new_bucket = storage_client.create_bucket(bucket, location="us-west1")

    print(
        "Created bucket {} in {} with storage class {}".format(
            new_bucket.name, new_bucket.location, new_bucket.storage_class
        )
    )
    return new_bucket

def export_dataset(dataset):
    # tables = list_tables(dataset)
    
    bucket = f'export_pqt_{dataset}_new'
    print (f'Exporting to bucket : {bucket}')
    client = storage.Client()
    if not Bucket(client, bucket).exists():
        create_bucket_class_location(bucket)
    
    tables = ["corpus_chains_2048_unique"]
    for table in tables:
        export_table(dataset, table, bucket)
    print(f'\n\nTo download bucket please run command below in terminal:\n\n\tgsutil -m cp -r gs://{bucket} .')

if __name__ == "__main__":
    dataset = sys.argv[2]
    export_dataset(dataset)