import cv2
import numpy as np

from ch6.q1 import BlurLibrary


def pepper_salt_noise(img, p=0.1, noise_type=0):
    img_noised = np.copy(img)
    num = int(np.around(img_noised.shape[0] * img_noised.shape[1] * p))
    noise_i_idx = np.random.choice(img_noised.shape[0], num)
    noise_j_idx = np.random.choice(img_noised.shape[1], num)
    if noise_type == 0:
        img_noised[noise_i_idx[:len(noise_i_idx) // 2], noise_j_idx[:len(noise_j_idx) // 2]] = 0
    elif noise_type == 1:
        img_noised[noise_i_idx[len(noise_i_idx) // 2:], noise_j_idx[len(noise_j_idx) // 2:]] = 255
    else:
        img_noised[noise_i_idx[:len(noise_i_idx) // 2], noise_j_idx[:len(noise_j_idx) // 2]] = 0
        img_noised[noise_i_idx[len(noise_i_idx) // 2:], noise_j_idx[len(noise_j_idx) // 2:]] = 255
    return img_noised


if __name__ == '__main__':
    lena = cv2.imread('images/lena.bmp', 0)
    lena_noised = pepper_salt_noise(lena, p=0.2)

    # 算数均值滤波
    lena_blur = BlurLibrary.mean_blur(lena_noised, 3)
    # 几何均值滤波
    lena_geomatric_blur = BlurLibrary.geometric_mean_blur(lena_noised, k=3)
    # 谐波均值滤波
    lena_harmonic_blur = BlurLibrary.harmonic_mean_blur(lena_noised, k=3)
    # 逆谐波滤波器
    lena_inverse_harmonic_blur = BlurLibrary.inverse_harmonic_blur(lena_noised, q=-1, k=3)

    # 中值滤波器
    lena_median_blur = BlurLibrary.median_blur(lena_noised, 3)
    # 中点滤波器
    lena_midpoint_blur = BlurLibrary.midpoint_blur(lena_noised, k=3)

    cv2.imshow('lena_blur', lena_blur)
    cv2.imshow('lena_geomatric_blur', lena_geomatric_blur)
    cv2.imshow('lena_harmonic_blur', lena_harmonic_blur)
    cv2.imshow('lena_inverse_harmonic_blur', lena_inverse_harmonic_blur)

    cv2.imshow('lena_median_blur', lena_median_blur)
    cv2.imshow('lena_midpoint_blur', lena_midpoint_blur)

    cv2.imshow('lena', lena)
    cv2.imshow('lena_noised', lena_noised)
    cv2.waitKey(0)
