import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy


def gaussian_noise(img, var=250, mean=0):
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    return np.uint8(np.clip(img + noise, 0, 255))


def draw_gaussian_noised_images():
    lena = cv2.imread('images/lena.bmp', 0)
    lena_noised = [lena]
    plt.figure(figsize=(12, 8))
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


def geometric_mean_blur(img, ksize=5):
    img = copy.deepcopy(img)
    img_expanded = cv2.copyMakeBorder(img, ksize // 2, ksize // 2, ksize // 2, ksize // 2,
                                      borderType=cv2.BORDER_REPLICATE)
    for i in range(ksize // 2, len(img_expanded) - ksize // 2, 1):
        for j in range(ksize // 2, len(img_expanded[i]) - ksize // 2, 1):
            sub_img = img_expanded[i - ksize // 2:i + ksize // 2 + 1, j - ksize // 2:j + ksize // 2 + 1]
            img[i - ksize // 2, j - ksize // 2] = np.uint8(
                np.prod(
                    np.float64(sub_img)
                ) ** (1 / ksize / ksize)
            )
    return img


def harmonic_mean_blur(img, ksize=5):
    img = copy.deepcopy(img)
    img_expanded = cv2.copyMakeBorder(img, ksize // 2, ksize // 2, ksize // 2, ksize // 2,
                                      borderType=cv2.BORDER_REPLICATE)
    for i in range(ksize // 2, len(img_expanded) - ksize // 2, 1):
        for j in range(ksize // 2, len(img_expanded[i]) - ksize // 2, 1):
            img[i - ksize // 2, j - ksize // 2] = np.uint8(
                ksize * ksize / np.sum(1 / np.float64(
                    img_expanded[i - ksize // 2:i + ksize // 2 + 1, j - ksize // 2:j + ksize // 2 + 1]
                ))
            )
    return img


def midpoint_blur(img, ksize=5):
    img = copy.deepcopy(img)
    img_expanded = cv2.copyMakeBorder(img, ksize // 2, ksize // 2, ksize // 2, ksize // 2,
                                      borderType=cv2.BORDER_REPLICATE)
    for i in range(ksize // 2, len(img_expanded) - ksize // 2, 1):
        for j in range(ksize // 2, len(img_expanded[i]) - ksize // 2, 1):
            sub_img = np.uint64(img_expanded[i - ksize // 2:i + ksize // 2 + 1, j - ksize // 2:j + ksize // 2 + 1])
            img[i - ksize // 2, j - ksize // 2] = np.uint8(
                1 / 2 * (np.max(sub_img) + np.min(sub_img))
            )
    return img


def inverse_harmonic_blur(img, q, ksize=5):
    img = copy.deepcopy(img)
    img_expanded = cv2.copyMakeBorder(img, ksize // 2, ksize // 2, ksize // 2, ksize // 2,
                                      borderType=cv2.BORDER_REPLICATE)
    for i in range(ksize // 2, len(img_expanded) - ksize // 2, 1):
        for j in range(ksize // 2, len(img_expanded[i]) - ksize // 2, 1):
            sub_img = np.float64(img_expanded[i - ksize // 2:i + ksize // 2 + 1, j - ksize // 2:j + ksize // 2 + 1])
            img[i - ksize // 2, j - ksize // 2] = np.sum(sub_img ** (q + 1)) / np.sum(sub_img ** q)
    return img


def recover():
    img = cv2.imread('images/lena.bmp', 0)
    img_noised = gaussian_noise(img)
    # 算数均值滤波
    img_blur = cv2.blur(copy.deepcopy(img_noised), (5, 5))
    # 几何均值滤波
    img_geomatric_blur = geometric_mean_blur(img_noised)
    # 谐波均值滤波
    img_harmonic_blur = harmonic_mean_blur(img_noised)
    # 逆谐波滤波器
    img_inverse_harmonic_blur = inverse_harmonic_blur(img_noised, q=-1)

    # 中值滤波器
    img_median_blur = cv2.medianBlur(copy.deepcopy(img_noised), 5)
    # 中点滤波器
    img_midpoint_blur = midpoint_blur(img_noised)

    cv2.imshow('img', img)
    cv2.imshow('noised', img_noised)
    cv2.imshow('img_blur', img_blur)
    cv2.imshow('img_geomatric_blur', img_geomatric_blur)
    cv2.imshow('img_harmonic_blur', img_harmonic_blur)
    cv2.imshow('img_inverse_harmonic_blur', img_inverse_harmonic_blur)

    cv2.imshow('img_median_blur', img_median_blur)
    cv2.imshow('img_midpoint_blur', img_midpoint_blur)

    cv2.waitKey(0)


if __name__ == '__main__':
    recover()
