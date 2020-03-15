import cv2
import numpy as np
import matplotlib.pyplot as plt

from hist import elain, lena


def local_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(7, 7))
    return clahe.apply(img)


if __name__ == '__main__':
    elain = elain[1]
    lena = lena[1]

    for img in [elain, lena]:
        img_eq = local_hist(img)
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Original image")
        plt.subplot(2, 2, 3)
        plt.hist(img.ravel(), 256, [0, 256])
        plt.subplot(2, 2, 2)
        plt.imshow(img_eq, cmap="gray")
        plt.title("Local histogram equalized image")
        plt.subplot(2, 2, 4)
        plt.hist(img_eq.ravel(), 256, [0, 256])
        plt.show()
