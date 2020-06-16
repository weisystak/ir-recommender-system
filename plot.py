import matplotlib.pyplot as plt

a = [16, 32, 64, 128]
b = [0.01904, 0.01656, 0.01247, 0.00722]

plt.scatter(a, b)
plt.xlabel('dim')
plt.ylabel('MAP')
plt.savefig('a.png')