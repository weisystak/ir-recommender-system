from svd import SVD
from util import *
import torch.optim as optim
import torch
from tqdm import tqdm

torch.cuda.set_device(1)
batch_size = 1024
dim = 32  #  16, 32, 64, 128
lr = 1E-2
lamb = 1E-3 # 1E-5 # regularization
n_epochs = 50
ng_factor = 1

data_path = "train.csv"
sub_csv = "submission.csv"
# ans_file = f"output_BCE/ans_dim{dim}_epoch_{n_epochs}.csv"
ans_file = f"output/ans_dim{dim}_epoch_{n_epochs}_ng_factor{ng_factor}.csv"
# ans_file = f"output/ans_dim{dim}_epoch_{n_epochs}_SGD_{lr}_lamb{lamb}.csv"

X, X_val, mat, n_user, n_item = load_data(data_path, ng_factor=ng_factor, need_val=True, val_rate=0.1)

train_data = Data(X, mat, n_item, ng=True)
val_data = Data(X_val)


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=1)


model = SVD(n_user, n_item, dim)
model.cuda()


# optimizer = optim.SGD( model.parameters(), lr=lr, weight_decay=lamb)
optimizer = optim.Adam(model.parameters())
model.train()
train_loss = 0
train_acc = 0
val_loss = 0
val_acc = 0
min_val_loss = 1E10
for epoch in range(n_epochs):
    for uid, iid, jid in tqdm(train_loader):
        optimizer.zero_grad()
        predi, predj = model(uid.cuda(), iid.cuda(), jid.cuda())
        
        # loss = BCELoss(predi, predj)
        loss = BPRLoss(predi, predj)
        # loss = ((1 - predi)**2).mean() + (predj**2).mean()

        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().item()
        train_acc += np.sum((predi.cpu().detach().numpy()) > 0) + np.sum((predj.cpu().detach().numpy()) <= 0)
    
    model.eval()
    for uid, iid, jid in val_loader:
        predi, predj = model(uid.cuda(), iid.cuda(), jid.cuda())
        
        # loss = BCELoss(predi, predj)
        # loss = BPRLoss(predi, predj)
        # loss = ((1 - predi)**2).mean() 


        # val_loss += loss.cpu().item()
        val_acc += np.sum((predi.cpu().detach().numpy()) > 0) # + np.sum(np.round(predj.cpu().detach().numpy()) == 0)

    train_acc = train_acc/len(train_data)/2
    train_loss = train_loss/len(train_data)/2
    val_acc = val_acc/len(val_data)
    # val_loss = val_loss/len(val_data)

    # if loss < min_val_loss:
    #     torch.save(model, f"ckpts/svd_{dim}.bin")
    #     min_val_loss = loss
    
    
    # print(f'{epoch:>4d}/{n_epochs:>4d}: train {train_acc:.3f} {train_loss:.3f} {val_acc:.3f} {val_loss:.3f}')
    print(f'{epoch+1:>4d}/{n_epochs:>4d}: train {train_acc:.3f} {train_loss:.3f} {val_acc:.3f}')
    # print(np.round(predi.cpu().detach().numpy()))
    # print(np.round(predj.cpu().detach().numpy()))
    



out = open( ans_file, "w")
out.write("UserId,ItemId\n")

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


