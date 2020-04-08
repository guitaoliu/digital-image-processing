import cv2
import copy
import numpy as np


class BlurLibrary:

    def __init__(self, img, k):
        self.img = img
        self.k = k

    def mean_blur(self):
        img = copy.deepcopy(self.img)
        return cv2.blur(img, ksize=(self.k, self.k))

    def geometric_mean_blur(self):
        img = copy.deepcopy(self.img)
        img_expanded = cv2.copyMakeBorder(img, self.k // 2, self.k // 2, self.k // 2, self.k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(self.k // 2, len(img_expanded) - self.k // 2, 1):
            for j in range(self.k // 2, len(img_expanded[i]) - self.k // 2, 1):
                sub_img = np.float64(img_expanded[
                                     i - self.k // 2:i + self.k // 2 + 1,
                                     j - self.k // 2:j + self.k // 2 + 1
                                     ])
                pixel = np.around(np.prod(sub_img) ** (1 / (self.k * self.k)))
                img[i - self.k // 2, j - self.k // 2] = np.uint8(pixel)
        return img

    def harmonic_mean_blur(self):
        img = copy.deepcopy(self.img)
        img_expanded = cv2.copyMakeBorder(img, self.k // 2, self.k // 2, self.k // 2, self.k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(self.k // 2, len(img_expanded) - self.k // 2, 1):
            for j in range(self.k // 2, len(img_expanded[i]) - self.k // 2, 1):
                img[i - self.k // 2, j - self.k // 2] = np.uint8(
                    self.k * self.k / np.sum(1 / np.float64(
                        img_expanded[i - self.k // 2:i + self.k // 2 + 1, j - self.k // 2:j + self.k // 2 + 1]
                    ))
                )
        return img

    def inverse_harmonic_blur(self, q):
        img = copy.deepcopy(self.img)
        img_expanded = cv2.copyMakeBorder(img, self.k // 2, self.k // 2, self.k // 2, self.k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(self.k // 2, len(img_expanded) - self.k // 2, 1):
            for j in range(self.k // 2, len(img_expanded[i]) - self.k // 2, 1):
                sub_img = np.float64(
                    img_expanded[i - self.k // 2:i + self.k // 2 + 1, j - self.k // 2:j + self.k // 2 + 1])
                img[i - self.k // 2, j - self.k // 2] = np.sum(sub_img ** (q + 1)) / np.sum(sub_img ** q)
        return img

    def min_blur(self):
        img = copy.deepcopy(self.img)
        img_expanded = cv2.copyMakeBorder(img, self.k // 2, self.k // 2, self.k // 2, self.k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(self.k // 2, len(img_expanded) - self.k // 2, 1):
            for j in range(self.k // 2, len(img_expanded[i]) - self.k // 2, 1):
                sub_img = np.uint64(
                    img_expanded[i - self.k // 2:i + self.k // 2 + 1, j - self.k // 2:j + self.k // 2 + 1])
                img[i - self.k // 2, j - self.k // 2] = np.uint8(
                    np.min(sub_img)
                )
        return img

    def max_blur(self):
        img = copy.deepcopy(self.img)
        img_expanded = cv2.copyMakeBorder(img, self.k // 2, self.k // 2, self.k // 2, self.k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(self.k // 2, len(img_expanded) - self.k // 2, 1):
            for j in range(self.k // 2, len(img_expanded[i]) - self.k // 2, 1):
                sub_img = np.uint64(
                    img_expanded[i - self.k // 2:i + self.k // 2 + 1, j - self.k // 2:j + self.k // 2 + 1])
                img[i - self.k // 2, j - self.k // 2] = np.uint8(
                    np.max(sub_img)
                )
        return img

    def median_blur(self):
        return cv2.medianBlur(copy.deepcopy(self.img), ksize=self.k)

    def midpoint_blur(self):
        img = copy.deepcopy(self.img)
        img_expanded = cv2.copyMakeBorder(img, self.k // 2, self.k // 2, self.k // 2, self.k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(self.k // 2, len(img_expanded) - self.k // 2, 1):
            for j in range(self.k // 2, len(img_expanded[i]) - self.k // 2, 1):
                sub_img = np.uint64(
                    img_expanded[i - self.k // 2:i + self.k // 2 + 1, j - self.k // 2:j + self.k // 2 + 1])
                img[i - self.k // 2, j - self.k // 2] = np.uint8(
                    1 / 2 * (np.max(sub_img) + np.min(sub_img))
                )
        return img

    def alpha_blur(self, d=2):
        img = copy.deepcopy(self.img)
        img_expanded = cv2.copyMakeBorder(img, self.k // 2, self.k // 2, self.k // 2, self.k // 2,
                                          borderType=cv2.BORDER_REPLICATE)
        for i in range(self.k // 2, len(img_expanded) - self.k // 2, 1):
            for j in range(self.k // 2, len(img_expanded[i]) - self.k // 2, 1):
                sub_img = np.uint64(
                    img_expanded[i - self.k // 2:i + self.k // 2 + 1, j - self.k // 2:j + self.k // 2 + 1])
                sorted_img = sorted(sub_img.flatten())
                img[i - self.k // 2, j - self.k // 2] = np.uint8(
                    1 / (sub_img.shape[0] * sub_img.shape[1] - d) * np.sum(sorted_img[d // 2:-d // 2])
                )
        return img
