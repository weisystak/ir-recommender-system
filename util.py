import re
import numpy as np
from functools import wraps
import time
from torch.utils.data import DataLoader, Dataset

def divide_data(X, val_rate):
        n = len(X)
        
        val_idx = np.random.choice(n, round(val_rate * n), replace=False)
        mask = np.ones( n, np.bool)
        mask[val_idx] = 0

        return X[mask], X[val_idx]

class Data(Dataset):
    def __init__(self, X, mat=None, n_item=None, ng=False):
        super(Data, self).__init__()
        self.X = X
        self.mat = mat
        self.ng = ng
        self.n_item = n_item

    def __len__(self):
        return self.X.shape[0]
    
    
    def __getitem__(self, idx):
        uid, iid, jid = self.X[idx]

        if self.ng:
            jid = np.random.randint(self.n_item)
            while jid in self.mat[uid]:
                jid = np.random.randint(self.n_item)
        return uid, iid, jid

def BCELoss(predi, predj):
        eps = 1e-20
        return -(predi.sigmoid().log().sum() + (1-predj.sigmoid()+eps).log().sum())

def BPRLoss(predi, predj):
    eps = 1e-20
    return -(predi-predj).sigmoid().log().sum()

def timer(title):
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            print(f'{title}: {end - start}s')

            return result
        return wrapper
    return decorator



def make_example(mat):
    X = []  
    for uid in range(mat.shape[0]):
        row = mat[uid]

        for iid in row:
            X.append( (uid, iid, -1) ) 

    X = np.array(X, dtype=np.int)      # ~= (309077 * (ng_factor+1), 3)
    return X

@timer(title="load data")
def load_data(data_path, ng_factor=0, need_val=False, val_rate=0.1): 
    train_file = open(data_path)
    # remove first line (UserId,ItemId)
    next(train_file)    
    max_uid = 0
    max_iid = 0

    mat = []
    for line in train_file:
        line = line.split()
        uid, line[0] = line[0].split(',')  # '0,1938'.split(',')
        max_uid = max(int(uid), max_uid)

        row = []
        for iid in line:
            iid = int(iid)
            row.append(iid)
            max_iid = max(iid, max_iid)
        mat.append(row)
    mat = np.array(mat)

    n_user = max_uid + 1
    n_item = max_iid + 1

    X = make_example(mat)
    if need_val:
        X, X_val = divide_data(X, val_rate)
    else:
        X_val = None

    X = np.tile(X, (ng_factor, 1))
    return X, X_val, mat, n_user, n_item
