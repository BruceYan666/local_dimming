from PIL import Image
import numpy as np
import math
from rgbToyuv import rgbToyuv
from getBL import getBL
from compute_objectives import compute_objectives
from get_mse import get_mse
from smoothBL_BMA import smoothBL_BMA
from two_stage_com import two_stage_com
from initialize_pop import initialize_pop
import random
import pdb

imgspath = './picture/9.jpg'
img = Image.open(imgspath)
img = np.array(img)
# print(img.shape)
# print(img.dtype)
# print(img)
img1 = np.double(img)
[Y,U,V] = rgbToyuv(img1)
m = np.size(Y, axis=0)
n = np.size(Y, axis=1)
BL = getBL(Y)
BL=BL.astype(int)
BL_NEW = BL.reshape(1, 36*66)
PC_LUT = sum(sum(BL))/255
# print(PC_LUT)
[MSE,PIC] = compute_objectives(BL_NEW,m,n,img,Y)
print('基准背光值的mse为%f\n'%MSE)

N = 10000
Data = initialize_pop(N,PC_LUT,BL_NEW)
c = np.size(Data, axis=1)
mse = np.zeros(N)
# pdb.set_trace()
for j in range(N):
    mse[j], _ = compute_objectives(Data[j, 0:c], m, n, img, Y)
# print(Data.shape)
Data = np.c_[Data, mse]
# print(Data.shape)
# pdb.set_trace()
x=Data[:,0:c]#切片操作
y=Data[:,c]
# print(x.shape)
# print(y.shape)
np.savetxt("./cache/9Data.txt", Data)
np.savetxt("./cache/9x.txt", x)
np.savetxt("./cache/9y.txt", y)
