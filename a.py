import torch

torch.manual_seed(0)

a = torch.tensor([1,4])
b = torch.tensor([2,1])
print(a@b)
n_user = 100
n_factors = 5
n_item = 1000

lr = 1E-5
reg = 5E-5

pu = torch.rand((n_user, n_factors), requires_grad=True)
qi = torch.rand((n_item, n_factors), requires_grad=True)

bu = torch.zeros(n_user, requires_grad=True)
bi = torch.zeros(n_item, requires_grad=True)


user, item, rating = 20, \
                     20, \
                     torch.tensor(0.8, requires_grad=False)


pred = bu[user] + bi[item]

pred += pu[user] @ qi[item]
# print(pu[user] @ qi[item])

err = rating - pred

err.backward()

# print(err.grad)
print(bu.grad[user])
print(bi.grad[item])
print(pu.grad[user])
print(qi.grad[item])


user, item, rating = 10, \
                     10, \
                     torch.tensor(0.8, requires_grad=False)

# bu.grad.zero_()
# err.grad.zero_()
pred = bu[user] + bi[item]

pred += pu[user] @ qi[item]
# print(pu[user] @ qi[item])

err = rating - pred

err.backward()


print(bu.grad[user])
print(bi.grad[item])
print(pu.grad[user])
print(qi.grad[item])
# print(pred.grad)

# Update biases
# bu[user] += lr * (err - reg * bu[user])
# bi[item] += lr * (err - reg * bi[item])

# # Update latent factors
# for factor in range(n_factors):
#     puf = pu[user, factor]
#     qif = qi[item, factor]

#     pu[user, factor] += lr * (err * qif - reg * puf)
#     qi[item, factor] += lr * (err * puf - reg * qif)