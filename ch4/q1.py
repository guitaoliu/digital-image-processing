import cv2
import numpy as np
import matplotlib.pyplot as plt

def median_blur(img):
    imgs = {}
    for i, n in enumerate([3, 5, 7]):
        imgs[i+1] = cv2.medianBlur(img, n)
    return imgs


def gaussian_blur(img):
    imgs = {}
    for i, n in enumerate([(3, 3), (5, 5), (7, 7)]):
        imgs[i+1] = cv2.GaussianBlur(img, n, 1.5)
    return imgs


def show_comparison(img_ori, imgs_blured):
    imgs = {0: img_ori}
    imgs.update(imgs_blured) 
    plt.figure()
    for i in range(len(imgs)):
        plt.subplot(2, 2, i+1)
        title = 'origin image' if i == 0 else str(2*i+1) + '×' + str(2*i+1) + ' blured'
        plt.title(title)
        plt.imshow(imgs[i], cmap='gray')
    plt.show()


if __name__ == '__main__':
    test1 = cv2.imread('images/test1.pgm', 0)
    # test1_median_blur = median_blur(test1)
    test1_gaussian_blur = gaussian_blur(test1)
    # show_comparison(test1, test1_median_blur)
    show_comparison(test1, test1_gaussian_blur)

    test2 = cv2.imread('images/test2.tif', 0)
    # test2_median_blur = median_blur(test2)
    test2_gaussian_blur = gaussian_blur(test2)
    # show_comparison(test2, test2_median_blur)
    show_comparison(test2, test2_gaussian_blur)
