def compute_objectives(D, m, n, img, Y):
    from smoothBL_BMA import smoothBL_BMA
    from two_stage_com import two_stage_com
    from get_mse import get_mse

    temp = D.reshape(36, 66)
    BL1_smooth = smoothBL_BMA(temp)
    for i in range(m):
        for j in range(n):
            if BL1_smooth[i,j] < 0:
                BL1_smooth[i,j] = 0
            if BL1_smooth[i,j] > 255:
                BL1_smooth[i,j] = 255

    [RGB1,Y_1]= two_stage_com(BL1_smooth, img)
    mse=get_mse(Y_1, Y)

    return mse, RGB1