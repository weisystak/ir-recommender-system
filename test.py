import torch
import sys
from util import *

data_path = "train.csv"
_, _, mat, n_user, n_item = load_data(data_path)

out = open( sys.argv[1], "w")
out.write("UserId,ItemId\n")

model = torch.load('ckpts/svd_16.bin')
res = model.p.weight @ model.q.weight.T

res = torch.argsort(res,  descending=True)
for uid in range(mat.shape[0]):
    out.write(f'{uid},')
    r = res[uid]

    positives = mat[uid]
    z = torch.zeros_like(r, dtype=torch.bool)
    for elem in positives:
        z = z | (r != elem)
    r = r[z][:50]

    for iid in r:
        out.write(f'{iid} ')
    out.write("\n")