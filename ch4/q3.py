import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask

def unsharp_masking(img, do_plt=True):
    img_out = unsharp_mask(img, radius=5)

    # img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    # img_out = cv2.addWeighted(img, 2, img_blur, -1, 0)
    if do_plt:
        plt.figure()
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title('origin image')
        plt.subplot(122)
        plt.title('enhanced image')
        plt.imshow(img_out, cmap='gray')
        plt.show()
    return img

def high_pass_filter(img):
    img_sobel_x_edge_detect = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_y_edge_detect = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel_edge_detect = cv2.convertScaleAbs(
        np.sqrt(img_sobel_x_edge_detect**2 + img_sobel_y_edge_detect**2)
    )

    img_laplace_edge_detect = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F, ksize=3))
    img_canny_edge_detect = cv2.convertScaleAbs(cv2.Canny(img, 50, 150))

    plt.figure()
    plt.subplot(131)
    plt.title('Sobel edge detect')
    plt.imshow(img_sobel_edge_detect, cmap='gray')
    plt.subplot(132)
    plt.title('Laplace edge detect')
    plt.imshow(img_laplace_edge_detect, cmap='gray')
    plt.subplot(133)
    plt.title('Canny edge detect')
    plt.imshow(img_canny_edge_detect, cmap='gray')
    plt.show()


if __name__ == '__main__':
    test3 = cv2.imread('images/test3_corrupt.pgm', cv2.IMREAD_GRAYSCALE)
    test4 = cv2.imread('images/test4.tif', cv2.IMREAD_GRAYSCALE)
    # unsharp_masking(test3)
    # unsharp_masking(test4)

    high_pass_filter(test3)
    high_pass_filter(test4)

