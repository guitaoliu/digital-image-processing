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

class BlurLibrary():

    @staticmethod
    def mean_blur(img, k=5):
        img = copy.deepcopy(img)
        return cv2.blur(img, ksize=(k, k))

    @staticmethod
    def geometric_mean_blur(img, k=5):
        img = copy.deepcopy(img)
        img_expanded = cv2.copyMakeBorder(img, k // 2, k // 2, k // 2, k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(k // 2, len(img_expanded) - k // 2, 1):
            for j in range(k // 2, len(img_expanded[i]) - k // 2, 1):
                sub_img = img_expanded[i - k // 2:i + k // 2 + 1, j - k // 2:j + k // 2 + 1]
                img[i - k // 2, j - k // 2] = np.uint8(
                    np.prod(
                        np.float64(sub_img)
                    ) ** (1 / k / k)
                )
        return img

    @staticmethod
    def harmonic_mean_blur(img, k=5):
        img = copy.deepcopy(img)
        img_expanded = cv2.copyMakeBorder(img, k // 2, k // 2, k // 2, k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(k // 2, len(img_expanded) - k // 2, 1):
            for j in range(k // 2, len(img_expanded[i]) - k // 2, 1):
                img[i - k // 2, j - k // 2] = np.uint8(
                    k * k / np.sum(1 / np.float64(
                        img_expanded[i - k // 2:i + k // 2 + 1, j - k // 2:j + k // 2 + 1]
                    ))
                )
        return img

    @staticmethod
    def inverse_harmonic_blur(img, q, k=5):
        img = copy.deepcopy(img)
        img_expanded = cv2.copyMakeBorder(img, k // 2, k // 2, k // 2, k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(k // 2, len(img_expanded) - k // 2, 1):
            for j in range(k // 2, len(img_expanded[i]) - k // 2, 1):
                sub_img = np.float64(img_expanded[i - k // 2:i + k // 2 + 1, j - k // 2:j + k // 2 + 1])
                img[i - k // 2, j - k // 2] = np.sum(sub_img ** (q + 1)) / np.sum(sub_img ** q)
        return img

    @staticmethod
    def min_blur(img, k=3):
        img = copy.deepcopy(img)
        img_expanded = cv2.copyMakeBorder(img, k // 2, k // 2, k // 2, k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(k // 2, len(img_expanded) - k // 2, 1):
            for j in range(k // 2, len(img_expanded[i]) - k // 2, 1):
                sub_img = np.uint64(img_expanded[i - k // 2:i + k // 2 + 1, j - k // 2:j + k // 2 + 1])
                img[i - k // 2, j - k // 2] = np.uint8(
                    np.min(sub_img)
                )
        return img

    @staticmethod
    def max_blur(img, k=3):
        img = copy.deepcopy(img)
        img_expanded = cv2.copyMakeBorder(img, k // 2, k // 2, k // 2, k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(k // 2, len(img_expanded) - k // 2, 1):
            for j in range(k // 2, len(img_expanded[i]) - k // 2, 1):
                sub_img = np.uint64(img_expanded[i - k // 2:i + k // 2 + 1, j - k // 2:j + k // 2 + 1])
                img[i - k // 2, j - k // 2] = np.uint8(
                    np.max(sub_img)
                )
        return img

    @staticmethod
    def median_blur(img, k=5):
        return cv2.medianBlur(copy.deepcopy(img), ksize=k)

    @staticmethod
    def midpoint_blur(img, k=5):
        img = copy.deepcopy(img)
        img_expanded = cv2.copyMakeBorder(img, k // 2, k // 2, k // 2, k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(k // 2, len(img_expanded) - k // 2, 1):
            for j in range(k // 2, len(img_expanded[i]) - k // 2, 1):
                sub_img = np.uint64(img_expanded[i - k // 2:i + k // 2 + 1, j - k // 2:j + k // 2 + 1])
                img[i - k // 2, j - k // 2] = np.uint8(
                    1 / 2 * (np.max(sub_img) + np.min(sub_img))
                )
        return img

    @staticmethod
    def alpha_blur(img, d=2, k=3):
        img = copy.deepcopy(img)
        img_expanded = cv2.copyMakeBorder(img, k // 2, k // 2, k // 2, k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(k // 2, len(img_expanded) - k // 2, 1):
            for j in range(k // 2, len(img_expanded[i]) - k // 2, 1):
                sub_img = np.uint64(img_expanded[i - k // 2:i + k // 2 + 1, j - k // 2:j + k // 2 + 1])
                sorted_img = sorted(sub_img.flatten())
                img[i - k // 2, j - k // 2] = np.uint8(
                    1 / (img_expanded.shape[0] * img_expanded.shape[1] - d) * np.sum(sorted_img[d // 2:-d // 2])
                )
        return img


def recover():
    img = cv2.imread('images/lena.bmp', 0)
    img_noised = gaussian_noise(img)
    # 算数均值滤波
    img_blur = BlurLibrary.mean_blur(img_noised, 5)
    # 几何均值滤波
    img_geomatric_blur = BlurLibrary.geometric_mean_blur(img_noised)
    # 谐波均值滤波
    img_harmonic_blur = BlurLibrary.harmonic_mean_blur(img_noised)
    # 逆谐波滤波器
    img_inverse_harmonic_blur = BlurLibrary.inverse_harmonic_blur(img_noised, q=-1)

    # 中值滤波器
    img_median_blur = BlurLibrary.median_blur(copy.deepcopy(img_noised), 5)
    # 中点滤波器
    img_midpoint_blur = BlurLibrary.midpoint_blur(img_noised)

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
