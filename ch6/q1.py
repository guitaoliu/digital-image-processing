import cv2
import matplotlib.pyplot as plt
import numpy as np

from ch6 import BlurLibrary


def gaussian_noise(img, var=250, mean=0):
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    return np.uint8(np.clip(img + noise, 0, 255))


def draw_gaussian_noised_images():
    lena = cv2.imread('images/lena.bmp', 0)
    lena_noised = [lena]
    plt.figure(figsize=(12, 8), dpi=400)
    plt.subplot(221)
    plt.imshow(lena, cmap='gray')
    plt.title('Original Image')
    for var in range(100, 400, 100):
        img = gaussian_noise(lena, var=var)
        lena_noised.append(img)
        plt.subplot(2, 2, var // 100 + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'var={var}, mean=0')
    plt.show()
    return lena_noised


def draw_comparing_img(img, noised, recovered):
    plt.figure(figsize=(12, 8), dpi=400)
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(noised, cmap='gray')
    plt.title('Noised Image')
    plt.subplot(133)
    plt.imshow(recovered, cmap='gray')
    plt.title('Recovered Image')
    plt.show()


def draw_recovers():
    img = cv2.imread('images/lena.bmp', 0)
    img_noised = gaussian_noise(img)
    recover_img_lib = BlurLibrary(img_noised, 3)
    draw_comparing_img(img, img_noised, recover_img_lib.mean_blur())
    draw_comparing_img(img, img_noised, recover_img_lib.geometric_mean_blur())
    draw_comparing_img(img, img_noised, recover_img_lib.harmonic_mean_blur())
    draw_comparing_img(img, img_noised, recover_img_lib.min_blur())
    draw_comparing_img(img, img_noised, recover_img_lib.max_blur())
    draw_comparing_img(img, img_noised, recover_img_lib.mean_blur())
    # draw_comparing_img(img, img_noised, recover_img_lib.inverse_harmonic_blur(q=1))
    draw_comparing_img(img, img_noised, recover_img_lib.median_blur())
    draw_comparing_img(img, img_noised, recover_img_lib.midpoint_blur())
    draw_comparing_img(img, img_noised, recover_img_lib.alpha_blur(d=4))


if __name__ == '__main__':
    draw_gaussian_noised_images()
    draw_recovers()
