from svd import *

data_path = "train.csv"
sub_csv = "submission.csv"

X, n_user, n_item = dataset(data_path, ng_factor=0)
svd = SVD()
svd.train(X, n_user, n_item)
# svd.test(csv=sub_csv)