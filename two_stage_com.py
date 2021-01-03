def two_stage_com(BL_smoothed, img):
    import numpy as np
    from rgbToyuv import rgbToyuv
    from yuvTorgb import yuvTorgb
    [Y, U, V] = rgbToyuv(img)
    img = np.double(img)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    BL_full = 255
    r = 2.2
    a = 0.005
    I_avg = (R + G + B) / 3

    Y1 = I_avg / (1 + np.exp(a * (BL_smoothed - Y)))
    k = (np.double(BL_smoothed) / BL_full) ** (1/r)
    np.seterr(invalid='ignore')
    Y2 = Y1 * np.log10(1 + k * Y)
    RGB1 = yuvTorgb(Y2, U, V)
    return RGB1, Y2