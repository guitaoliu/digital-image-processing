import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

from ch6 import BlurLibrary


def pepper_salt_noise(img, p=0.1):
    img_salt = copy.deepcopy(img)
    img_pepper = copy.deepcopy(img)
    img_salt_pepper = copy.deepcopy(img)

    num = int(np.around(img.shape[0] * img.shape[1] * p))
    noise_i_idx = np.random.choice(img.shape[0], num)
    noise_j_idx = np.random.choice(img.shape[1], num)

    img_pepper[noise_i_idx[:len(noise_i_idx) // 2], noise_j_idx[:len(noise_j_idx) // 2]] = 0

    img_salt[noise_i_idx[len(noise_i_idx) // 2:], noise_j_idx[len(noise_j_idx) // 2:]] = 255

    img_salt_pepper[noise_i_idx[:len(noise_i_idx) // 2], noise_j_idx[:len(noise_j_idx) // 2]] = 0
    img_salt_pepper[noise_i_idx[len(noise_i_idx) // 2:], noise_j_idx[len(noise_j_idx) // 2:]] = 255

    return img_salt, img_pepper, img_salt_pepper


def draw_comparison_plot(origin_img, blur_salt_img, blur_pepper_img, blur_salt_pepper_img):
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(origin_img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(222)
    plt.imshow(blur_salt_img, cmap='gray')
    plt.title('Added salt noise')
    plt.subplot(223)
    plt.imshow(blur_pepper_img, cmap='gray')
    plt.title('Added pepper noise')
    plt.subplot(224)
    plt.imshow(blur_salt_pepper_img, cmap='gray')
    plt.title('Added salt and pepper noise')
    plt.show()


def different_blur_function_test(lena, lena_salt_blur, lena_pepper_blur, lena_salt_pepper_blur):
    draw_comparison_plot(
        lena,
        lena_salt_blur.mean_blur(),
        lena_pepper_blur.mean_blur(),
        lena_salt_pepper_blur.mean_blur(),
    )

    draw_comparison_plot(
        lena,
        lena_salt_blur.geometric_mean_blur(),
        lena_pepper_blur.geometric_mean_blur(),
        lena_salt_pepper_blur.geometric_mean_blur(),
    )

    draw_comparison_plot(
        lena,
        lena_salt_blur.harmonic_mean_blur(),
        lena_pepper_blur.harmonic_mean_blur(),
        lena_salt_pepper_blur.harmonic_mean_blur(),
    )

    q = 1
    draw_comparison_plot(
        lena,
        lena_salt_blur.inverse_harmonic_blur(q),
        lena_pepper_blur.inverse_harmonic_blur(q),
        lena_salt_pepper_blur.inverse_harmonic_blur(q),
    )

    draw_comparison_plot(
        lena,
        lena_salt_blur.median_blur(),
        lena_pepper_blur.median_blur(),
        lena_salt_pepper_blur.median_blur(),
    )

    draw_comparison_plot(
        lena,
        lena_salt_blur.max_blur(),
        lena_pepper_blur.max_blur(),
        lena_salt_pepper_blur.max_blur(),
    )

    draw_comparison_plot(
        lena,
        lena_salt_blur.min_blur(),
        lena_pepper_blur.min_blur(),
        lena_salt_pepper_blur.min_blur(),
    )

    draw_comparison_plot(
        lena,
        lena_salt_blur.midpoint_blur(),
        lena_pepper_blur.midpoint_blur(),
        lena_salt_pepper_blur.midpoint_blur(),
    )

    d = 4
    draw_comparison_plot(
        lena,
        lena_salt_blur.alpha_blur(d),
        lena_pepper_blur.alpha_blur(d),
        lena_salt_pepper_blur.alpha_blur(d),
    )


if __name__ == '__main__':
    lena = cv2.imread('images/lena.bmp', 0)
    lena_salt, lena_pepper, lena_salt_pepper = pepper_salt_noise(lena, p=0.2)
    lena_salt_blur = BlurLibrary(lena_salt, 3)
    lena_pepper_blur = BlurLibrary(lena_pepper, 3)
    lena_salt_pepper_blur = BlurLibrary(lena_salt_pepper, 3)

    draw_comparison_plot(
        lena,
        lena_salt,
        lena_pepper,
        lena_salt_pepper
    )

    different_blur_function_test(lena, lena_salt_blur, lena_pepper_blur, lena_salt_pepper_blur)

    for q in range(-1, 2, 1):
        draw_comparison_plot(
            lena,
            lena_salt_blur.inverse_harmonic_blur(q),
            lena_pepper_blur.inverse_harmonic_blur(q),
            lena_salt_pepper_blur.inverse_harmonic_blur(q),
        )
