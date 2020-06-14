from util import *
import torch
from tqdm import tqdm


class SVD():
    def __init__(self, lr=1E-3, reg=0, n_epochs=100, dim=10):
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.dim = dim

    

    def _init_weights(self, n_user, n_item):
        self.n_user = n_user
        self.n_item = n_item

        self.p = torch.rand((n_user, self.dim), requires_grad=False)
        self.q = torch.rand((n_item, self.dim), requires_grad=False)
        self.bu = torch.zeros(n_user, requires_grad=False) # user bias
        self.bi = torch.zeros(n_item, requires_grad=False) # item bias
    

    def _divide_data(self, X, val_rate):
        n = len(X)
        
        val_idx = np.random.choice(n, round(val_rate * n), replace=False)
        mask = np.ones( n, np.bool)
        mask[val_idx] = 0

        return X[mask], X[val_idx]
    
    def _run_epoch(self, X, X_val):
        x_loss = 0
        x_acc = 0
        val_loss = 0
        val_acc = 0
        for i in tqdm(range(X.shape[0])):
            uid, iid, r = X[i]
            # print(uid, iid, r)
            pred = self.bu[uid] + self.bi[iid]
            pred += self.p[uid] @ self.q[iid]

            loss = r - pred

            self.bu[uid] += self.lr * (loss - self.reg * self.bu[uid])
            self.bi[iid] += self.lr * (loss - self.reg * self.bi[iid])
            self.p[uid]  += self.lr * (loss * self.q[iid] - self.reg * self.p[uid])
            self.q[iid]  += self.lr * (loss * self.p[uid] - self.reg * self.q[iid])

            x_loss += loss**2
            x_acc += 1 if r < 0.5 and pred < 0.5 or r >= 0.5 and pred >= 0.5 else 0

            
        for i in tqdm(range(X_val.shape[0])):
            uid, iid, r = X_val[i]
            pred = self.bu[uid] + self.bi[iid]
            pred += self.p[uid] @ self.q[iid]

            loss = r - pred

            val_loss += loss**2
            val_acc += 1 if r < 0.5 and pred < 0.5 or r >= 0.5 and pred >= 0.5 else 0
        return  x_acc/len(X), x_loss/len(X), val_acc/len(X_val), val_loss/len(X_val)
    
    def _early_stop(self, val_loss_list, delta=1E-2):  
        if val_loss_list[-1] + delta > val_loss_list[-2]:
            return True
        return False

    @timer(title="training")
    def train(self, X, n_user, n_item, val_rate=0.1):
        self._init_weights(n_user, n_item)
        X, X_val = self._divide_data(X, val_rate)
        
        val_loss_list = [40]
        for i in range(self.n_epochs):
            np.random.shuffle(X)

            train_acc, train_loss, val_acc, val_loss = self._run_epoch(X, X_val)

            print(f'{i:>4d}/{self.n_epochs:>4d}: train {train_acc:.3f} {train_loss:.3f} {val_acc:.3f} {val_loss:.3f}')
            val_loss_list.append(val_loss)
            if self._early_stop(val_loss_list):
                break

        return

    @timer(title="testing")
    def test(self, csv="submission.csv"):

        return
