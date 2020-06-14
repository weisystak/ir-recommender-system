import re
import numpy as np
from functools import wraps
import time

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

@timer(title="load data")
def dataset(data_path, ng_factor=0): 
    negative_sampling = True


    train_file = open(data_path)

    # remove first line (UserId,ItemId)
    next(train_file)    

    X = []
    ng = []



    max_uid = 0
    max_iid = 0


    for line in train_file:
        line = line.split()
        uid, line[0] = line[0].split(',')  # '0,1938'.split(',')
        max_uid = max(int(uid), max_uid)

        ngr = set()
        for iid in line:
            # [uid, iid, rating]
            X.append((int(uid), int(iid), 1.0))

            ngr.add(int(iid))
            max_iid = max(int(iid), max_iid)
        ng.append(ngr)
        

    X = np.array(X)     # (309077, 3)

    n_user = max_uid + 1
    n_item = max_iid + 1

    ngX = []
    if ng_factor:
        for uid, ngr in enumerate(ng):
            n = ng_factor * len(ngr)
            cnt = 0

            while cnt < n:
                iid = np.random.randint(n_item)
                if iid not in ngr:
                    ngX.append((uid, iid, 0.0))
                    cnt += 1
                else:
                    continue
        X = np.concatenate((X, ngX))    # (618154, 3)
    X = X.astype(np.int)
    return X, n_user, n_item






if __name__ == "__main__":
    dataset()