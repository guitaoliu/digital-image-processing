import numpy as np
import matplotlib.pyplot as plt
import cv2

from hist import citywall, lena, elain, women


def local_hist_enhance(img, e=3, k0=0.35, k1=0.02, k2=0.4):
    mean = np.mean(img)
    std = np.std(img)
    img_border = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    img_out = img_border.copy()
    for i in range(len(img_border))[1:-1]:
        for j in range(len(img_border))[1:-1]:
            img_local = img_border[i-1:i+2, j-1:j+2]
            mean_local = np.mean(img_local)
            std_local = np.std(img_local)
            if mean_local <= k0 * mean and k1 * std <= std_local <= k2 * std:
                img_out[i, j] = e * img_border[i, j]
    return img_out[1:-1, 1:-1]


if __name__ == '__main__':
    for img in [elain[1], lena[1]]:
        img_ehcd = local_hist_enhance(img)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.subplot(2, 2, 3)
        plt.hist(img.ravel(), 256, [0, 256])
        plt.subplot(2, 2, 2)
        plt.imshow(local_hist_enhance(img), cmap="gray")
        plt.subplot(2, 2, 4)
        plt.hist(img_ehcd.ravel(), 256, [0, 256])
        plt.show()
