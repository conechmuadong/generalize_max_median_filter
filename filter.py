import numpy as np


def generalize_max_median_filter(image, n, r):
    """"""
    image = np.array(image)
    image = image.astype(np.float64)
    rows, cols = image.shape
    new_image = np.zeros((rows, cols))

    image = np.pad(image, n, mode='edge')

    rows, cols = image.shape

    for i in range(n,rows-n):
        for j in range(n, cols-n):
            w1 = np.zeros(2*n)
            w2 = np.zeros(2*n)
            w3 = np.zeros(2*n)
            w4 = np.zeros(2*n)

            for k in range(-n, n):
                if k < 0:
                    w1[k+n] = image[i, j+k]
                    w2[k+n] = image[i+k, j+k]
                    w3[k+n] = image[i+k, j]
                    w4[k+n] = image[i-k, j+k]
                elif k > 0:
                    w1[k+n-1] = image[i, j+k]
                    w2[k+n-1] = image[i+k, j+k]
                    w3[k+n-1] = image[i+k, j]
                    w4[k+n-1] = image[i-k, j+k]
                else:
                    continue

            w1 = np.sort(w1)
            w2 = np.sort(w2)
            w3 = np.sort(w3)
            w4 = np.sort(w4)

            s1 = np.max(np.array([w1[r-1], w2[r-1], w3[r-1], w4[r-1]]))
            s2 = np.max(np.array([w1[2*n-r], w2[2*n-r], w3[2*n-r], w4[2*n-r]]))

            new_image[i-n, j-n] = np.median(np.array([s1, s2, image[i, j]]))

    return new_image

