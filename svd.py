import torch
import torch.nn as nn

class SVD(nn.Module):
    def __init__(self, n_user, n_item, dim):
        super(SVD, self).__init__()
        self.p = nn.Embedding(n_user, dim)
        self.q = nn.Embedding(n_item, dim)
        nn.init.normal_(self.p.weight, std=0.01)
        nn.init.normal_(self.q.weight, std=0.01)

    def forward(self, uid, iid, jid):
        user = self.p(uid)
        predi = (user * self.q(iid)).sum(dim=1)
        if jid[0] != -1: 
            predj = (user * self.q(jid)).sum(dim=1)
        else:
            predj = None
        
        return predi, predj
        