def rgbToyuv(img):
    import numpy as np
    img = np.double(img)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    Y = 0.2989 * R + 0.5866 * G + 0.1145 * B
    U = -0.1688 * R - 0.3312 * G + 0.5 * B + 128
    V = 0.5 * R - 0.4184 * G - 0.0816 * B + 128
    m = np.size(R, axis = 0)
    n = np.size(R, axis = 1)
    for i in range(m):
        for j in range(n):
            if Y[i,j] <0:
                Y[i,j]=0
            if Y[i,j] >255:
                Y[i,j]=255
            if U[i,j] <0:
                U[i,j]=0
            if U[i,j] >255:
                U[i,j]=255
            if V[i,j] <0:
                V[i,j]=0
            if V[i,j] >255:
                V[i,j]=255
    return Y, U, V