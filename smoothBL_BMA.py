def smoothBL_BMA(BL):
    import numpy as np
    from imresize import imresize
    K = 4
    a = 0.38
    b = 0.11
    c = 0.08
    d = 0.06

    for k in range(K):
        M = np.size(BL, axis=0)
        N = np.size(BL, axis=1)
        BL1 = np.zeros((M + 2, N + 2))
        BL2 = np.zeros((M + 2, N + 2))
        #镜像
        BL1[1:M+1, 1:N+1] = BL  #切片最后一位不算入

        BL1[0, 1:N+1] = BL[0, 0:N]
        BL1[M+1, 1:N+1] = BL[M-1, 0:N]

        BL1[1:M+1, 0] = BL[0:M, 0]
        BL1[1:M+1, N + 1] = BL[0:M, N - 1]

        BL1[0, 0] = BL[0, 0]
        BL1[0, N + 1] = BL[0, N - 1]
        BL1[M + 1, 0] = BL[M - 1, 0]
        BL1[M + 1, N + 1] = BL[M - 1, N - 1]
        #混光
        BL2[0:M+2, 0] = BL1[0:M+2, 0]
        BL2[0:M+2, N + 1] = BL1[0:M+2, N + 1]
        BL2[0, 0:N+2] = BL1[0, 0:N+2]
        BL2[M + 1, 0:N+2] = BL1[M + 1, 0:N+2]
        dd = BL1[0:M, 0:N] + BL1[0:M, 2:N+2] + BL1[2:M+2, 0:N] + BL1[2:M+2, 2:N+2]
        BL2[1:M+1, 1:N+1] = a * BL1[1:M+1, 1:N+1] + b * (BL1[1:M+1, 0:N] + BL1[1:M+1, 2:N+2]) + c * (BL1[0:M, 1:N+1] + BL1[2:M+2, 1:N+1]) + d * dd
        #插值
        BL2 = imresize(BL2, 2, 'bilinear')
        BL = BL2
    BL2 = imresize(BL2, 2, 'bicubic')
    return BL2


