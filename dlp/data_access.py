import pyarrow.parquet as pq
import pickle
import random

class PQDataAccess():
    def __init__(self, address, batch_size):
        self.batch_size = batch_size
                
        with open (address + '_info', 'rb') as fp:
            self.files_info = pickle.load(fp)
            random.shuffle(self.files_info)
        
        self.iterator = self.create_iterator()
          
    def create_iterator(self):
        temp_list = []
        for uri in self.files_info:
            uri = uri['uri']
            # /home/pooriya/Alireza/TED/data/export_pqt_0_ted/corpus_chains_sequence/000000000000
            new_uri = "/home/pooriya/Alireza/TED/data/export_pqt_0_ted/corpus_chains_sequence/" + uri.split('/')[-1]
            table = pq.read_table(new_uri).to_pandas()
            for index, row in table.iterrows():
                temp_list.append(row)
                if len(temp_list) == self.batch_size:
                    new_list = list(temp_list)
                    temp_list=[]
                    yield new_list

    def get_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            return None