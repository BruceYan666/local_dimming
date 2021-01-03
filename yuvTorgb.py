def yuvTorgb(Y,U,V):
    import numpy as np
    from PIL import Image
    import cv2
    R = Y + 1.4021 * (V - 128)
    G = Y - 0.3456 * (U - 128) - 0.7145 * (V - 128)
    B = Y + 1.771 * (U - 128)

    m = np.size(R, axis = 0)
    n = np.size(R, axis = 1)
    for i in range(m):
        for j in range(n):
            if R[i,j] <0:
                R[i,j]=0
            if R[i,j] >255:
                R[i,j]=255
            if G[i,j] <0:
                G[i,j]=0
            if G[i,j] >255:
                G[i,j]=255
            if B[i,j] <0:
                B[i,j]=0
            if B[i,j] >255:
                B[i,j]=255
    R = R.astype(np.uint8)
    G = G.astype(np.uint8)
    B = B.astype(np.uint8)
    # image = Image.merge("RGB", (R, G, B))
    image = cv2.merge([R, G, B])
    return image