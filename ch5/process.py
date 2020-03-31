import cv2
import matplotlib.pyplot as plt
import numpy as np

from ch5.filter import gaussian, butterworth, laplace, unsharp


def low_filter(img, radius=50):
    p, q = img.shape[0] * 2, img.shape[1] * 2
    img = cv2.copyMakeBorder(img, 0, img.shape[0], 0, img.shape[1], borderType=cv2.BORDER_CONSTANT, value=0)
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    gaussian_low_filter, _ = gaussian(p, q, radius=radius)
    butterworth_low_filter, _ = butterworth(p, q, radius=radius)
    gaussian_low = gaussian_low_filter * img_fft_shift
    butterworth_low = butterworth_low_filter * img_fft_shift
    img_gaussian_low = np.abs(np.fft.ifft2(np.fft.ifftshift(gaussian_low)))
    img_butterworth_low = np.abs(np.fft.ifft2(np.fft.ifftshift(butterworth_low)))
    p1 = np.sum(np.abs(gaussian_low) ** 2) / np.sum(np.abs(img_fft_shift) ** 2)
    p2 = np.sum(np.abs(butterworth_low) ** 2) / np.sum(np.abs(img_fft_shift) ** 2)
    return img_gaussian_low[:p // 2, :q // 2], p1, img_butterworth_low[:p // 2, :q // 2], p2


def high_filter(img, radius=50):
    p, q = img.shape[0] * 2, img.shape[1] * 2
    img = cv2.copyMakeBorder(img, 0, img.shape[0], 0, img.shape[1], borderType=cv2.BORDER_CONSTANT, value=0)
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    _, gaussian_high_filter = gaussian(p, q, radius=radius)
    _, butterworth_high_filter = butterworth(p, q, radius=radius)
    gaussian_high = gaussian_high_filter * img_fft_shift
    butterworth_high = butterworth_high_filter * img_fft_shift
    img_gaussian_high = np.abs(np.fft.ifft2(np.fft.ifftshift(gaussian_high)))
    img_butterworth_high = np.abs(np.fft.ifft2(np.fft.ifftshift(butterworth_high)))
    p1 = np.sum(np.abs(gaussian_high) ** 2) / np.sum(np.abs(img_fft_shift) ** 2)
    p2 = np.sum(np.abs(butterworth_high) ** 2) / np.sum(np.abs(img_fft_shift) ** 2)

    return img_gaussian_high[:p // 2, :q // 2], p1, img_butterworth_high[:p // 2, :q // 2], p2


def gaussian_butterworth():
    test1 = cv2.imread('imgaes/test1.pgm', 0)
    test2 = cv2.imread('imgaes/test2.tif', 0)
    test3 = cv2.imread('imgaes/test3_corrupt.pgm', 0)
    test4 = cv2.imread('imgaes/test4 copy.bmp', 0)

    image = test4
    result_gaussian = []
    result_butterworth = []
    for radius in range(25, 100, 25):
        g, p1, b, p2 = high_filter(image, radius)
        result_gaussian.append((g, p1))
        result_butterworth.append((b, p2))

    img = result_butterworth
    plt.figure()
    plt.subplot(221)
    plt.imshow(image, cmap='gray')
    plt.title('Origin image')
    for i, img in enumerate(img):
        plt.subplot(2, 2, i + 2)
        plt.imshow(img[0], cmap='gray')
        plt.title('radius={}, P={:.2%}'.format(25 * (i + 1), img[1]))
    plt.show()

    img = result_gaussian
    plt.figure()
    plt.subplot(221)
    plt.imshow(image, cmap='gray')
    plt.title('Origin image')
    for i, img in enumerate(img):
        plt.subplot(2, 2, i + 2)
        plt.imshow(img[0], cmap='gray')
        plt.title('radius={}, P={:.2%}'.format(25 * (i + 1), img[1]))
    plt.show()


def laplace_filter(img):
    p, q = img.shape[0] * 2, img.shape[1] * 2
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64FC1)
    img = cv2.copyMakeBorder(img, 0, img.shape[0], 0, img.shape[1], borderType=cv2.BORDER_CONSTANT, value=0)
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    laplace_mask = laplace(p, q)
    laplace_result = img_fft_shift * laplace_mask
    img_laplace = np.real(np.fft.ifft2(np.fft.ifftshift(laplace_result)))
    img_HPF = img_laplace[:p // 2, :q // 2]
    img_HPF_norm = img_HPF / np.max(np.abs(img_HPF))
    return img_HPF_norm[2:p // 2 - 2, 2:q // 2 - 2]
    # return img[:p//2, :q//2] - img_HPF_norm


def laplace_image_process():
    test3 = cv2.imread('imgaes/test3_corrupt.pgm', 0)
    test4 = cv2.imread('imgaes/test4 copy.bmp', 0)
    # test3_laplace = cv2.convertScaleAbs(laplace_filter(test3), alpha=255)
    test4_laplace = cv2.convertScaleAbs(laplace_filter(test4), alpha=255)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(test3, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(test3_laplace, cmap='gray')
    # plt.show()

    plt.figure()
    plt.subplot(121)
    plt.imshow(test4, cmap='gray')
    plt.subplot(122)
    plt.imshow(test4_laplace, cmap='gray')
    plt.show()



def unsharp_filter(img):
    p, q = img.shape[0] * 2, img.shape[1] * 2
    img = cv2.copyMakeBorder(img, 0, img.shape[0], 0, img.shape[1], borderType=cv2.BORDER_CONSTANT, value=0)
    img_fft_shift = np.fft.fftshift(np.fft.fft2(img))
    unsharp_mask = unsharp(p, q)
    img_unsharp = np.abs(
        np.fft.ifft2(
            np.fft.ifftshift(img_fft_shift * unsharp_mask))
    )
    return img_unsharp[:p // 2, :q // 2]


def unsharp_process():
    test3 = cv2.imread('imgaes/test3_corrupt.pgm', 0)
    test4 = cv2.imread('imgaes/test4 copy.bmp', 0)
    test3_unsharp = unsharp_filter(test3)
    test4_unsharp = unsharp_filter(test4)
    plt.figure()
    plt.subplot(121)
    plt.imshow(test3, cmap='gray')
    plt.subplot(122)
    plt.imshow(test3_unsharp, cmap='gray')
    plt.show()
    plt.figure()
    plt.subplot(121)
    plt.imshow(test4, cmap='gray')
    plt.subplot(122)
    plt.imshow(test4_unsharp, cmap='gray')
    plt.show()


if __name__ == '__main__':
    # gaussian_butterworth()
    # laplace_image_process()
    unsharp_process()
