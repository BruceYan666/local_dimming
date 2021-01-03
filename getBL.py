import numpy as np
import math

def getBL_MEAN(I):
    return np.mean(I)

def getBL_MAX(I):
    return np.max(I)

def getBL_LUT(I):
    avg = getBL_MEAN(I)
    max = getBL_MAX(I)
    Diff = max - avg
    cor = 0.5 * (Diff + (Diff ** 2) / 255)
    return avg + cor

def getBL(Y):
    I = np.around(Y)
    I = np.array(I)
    # 输入灰度图大小m*n
    m = np.size(Y, axis = 0)
    n = np.size(Y, axis = 1)
    # 得到区域图像大小h*w
    h = np.around(m / 36)
    w = np.around(n / 66)
    h = int(h)
    w = int(w)
    BL_LUT = np.zeros([36, 66])
    BL_MAX = np.zeros([36, 66])
    BL_MEAN = np.zeros([36, 66])

    # BL为36*66背光矩阵，每个背光值也叫分区亮度值，传统方法的值为LUT,MAX和MEAN
    for i in range(35):
        for j in range(65):
            x1 = h * i
            x2 = h * (i + 1) - 1
            y1 = w * j
            y2 = w * (j + 1) - 1
            C = I[x1:x2, y1:y2]
            BL_LUT[i ,j] = getBL_LUT(C)
            BL_MAX[i ,j] = getBL_MAX(C)
            BL_MEAN[i ,j] = getBL_MEAN(C)

    for j in range(65):
        x1 = 35 * h
        x2 = m - 1
        y1 = w * j
        y2 = w * (j + 1) - 1
        C = I[x1:x2, y1:y2]
        BL_LUT[35, j] = getBL_LUT(C)
        BL_MAX[35, j] = getBL_MAX(C)
        BL_MEAN[35, j] = getBL_MEAN(C)

    for i in range(35):
        x1 = h * i
        x2 = h * (i + 1) - 1
        y1 = w * 65
        y2 = n - 1
        C = I[x1:x2, y1:y2]
        BL_LUT[i, 65] = getBL_LUT(C)
        BL_MAX[i, 65] = getBL_MAX(C)
        BL_MEAN[i, 65] = getBL_MEAN(C)

    x1 = 35 * h
    x2 = m - 1
    y1 = 65 * w
    y2 = n - 1
    C = I[x1:x2, y1: y2]
    BL_LUT[35, 65] = getBL_LUT(C)
    BL_MAX[35, 65] = getBL_MAX(C)
    BL_MEAN[35, 65] = getBL_MEAN(C)
    return BL_LUT
