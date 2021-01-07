import torch
import numpy as np
import pdb
x = np.loadtxt('./cache/7x.txt')
m = np.size(x, axis=0)
n = np.size(x, axis=1)

for i in range(m):
    for j in range(n):
        if x[i][j] < 0:
            x[i][j] = 0
        elif x[i][j] > 255:
            x[i][j] = 255
pdb.set_trace()
np.savetxt("./cache/7X.txt", x)