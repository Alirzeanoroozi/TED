from pathlib import Path
import os
import pickle
import sys

def create_index(path):
    print(f'INDEXING DATASET {path}\t STARTED')
    
    files = [str(Path(path) / f) for f in os.listdir(path) if len(f) == 12 and f.startswith('0000')]
    
    with open(path + '_info.pkl', 'wb') as fp:
        pickle.dump(files, fp)
                
    print(f'INDEXING DATASET {path}\t FINISHED')
    
def index_dataset(dataset_folder):           
    for folder in [dataset_folder + x for x in os.listdir(dataset_folder) if os.path.isdir(dataset_folder + x)]:
        create_index(folder)
    
if __name__ == "__main__":
    args = sys.argv[1]
    index_dataset(args)    