def initialize_pop(GROUP,PC_MAX,BL_NEW):
    import numpy as np
    import pdb
    r = 20
    BL_UP = BL_NEW + r
    BL_LP = BL_NEW - r
    # pdb.set_trace()
    for i in range(len(BL_NEW)):
        if BL_UP[0, i] > 255:
            BL_UP[0, i] = 255
            BL_LP[0, i] = 2 * BL_NEW[0, i] - 255
        if BL_LP[0, i] < 0:
            BL_LP[0, i] = 0
            BL_UP[0, i] = 2 * BL_NEW[0, i]
    # pdb.set_trace()
    BL_UP1 = np.tile((BL_UP), (GROUP, 1))
    BL_LP1 = np.tile((BL_LP), (GROUP, 1))
    Data = np.zeros((GROUP, 36*66))
    for i in range(GROUP):
        PC = PC_MAX + 1
        while PC > PC_MAX:
            R = np.random.rand(1, 36*66)
            Data[i, :] = (BL_LP1[i, :] + (BL_UP1[i, :] - BL_LP1[i, :]) * R).astype(int)
            PC = np.sum(Data[i, :]) / 255

    return Data