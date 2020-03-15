import cv2
import matplotlib.pyplot as plt
import numpy as np


def hist(imgs):
    plt.figure(imgs[0])
    n = 0
    for img in imgs[1:]:
        plt.subplot(len(imgs), 1, n+1)
        plt.hist(img.ravel(), 256, [0, 256])
        plt.title(imgs[0] + str(n+1))
        n += 1

    plt.show()


def color_map_correct(img):
    return np.where(img == 0, 255, img)


citywall = [
    'citywall',
    cv2.imread('data/citywall.bmp', 0),
    cv2.imread('data/citywall1.bmp', 0),
    cv2.imread('data/citywall2.bmp', 0),
]
elain = [
    'elain',
    cv2.imread('data/elain.bmp', 0),
    cv2.imread('data/elain1.bmp', 0),
    cv2.imread('data/elain2.bmp', 0),
    cv2.imread('data/elain3.bmp', 0),
]
lena = [
    'lena',
    cv2.imread('data/lena.bmp', 0),
    cv2.imread('data/lena1.bmp', 0),
    cv2.imread('data/lena2.bmp', 0),
    cv2.imread('data/lena4.bmp', 0),
]
women = [
    'women',
    cv2.imread('data/woman.BMP', 0),
    cv2.imread('data/woman1.BMP', 0),
    cv2.imread('data/woman2.BMP', 0),
]

elain[2] = color_map_correct(elain[2])
citywall[2] = color_map_correct(citywall[2])
lena[2] = color_map_correct(lena[2])
women[2] = color_map_correct(women[2])


if __name__ == '__main__':
    hist(citywall)
    hist(elain)
    hist(lena)
    hist(women)


