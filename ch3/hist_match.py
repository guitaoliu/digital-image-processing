import numpy as np
import cv2
import matplotlib.pyplot as plt
from hist import citywall, lena, elain, women


def match(img_ori, img_ref):
    hist_ori = cv2.calcHist([img_ori], [0], None, [256], [0, 255])
    hist_ori = hist_ori / np.sum(hist_ori)
    hist_ref = cv2.calcHist([img_ref], [0], None, [256], [0, 255])
    hist_ref = hist_ref / np.sum(hist_ref)
    sum_ori = np.cumsum(hist_ori)
    sum_ref = np.cumsum(hist_ref)
    img_out = np.zeros_like(img_ori)

    for i in range(len(hist_ori)):
        tmp = np.abs(sum_ori[i] - sum_ref)
        smallest_idx = np.argwhere(tmp == np.min(tmp))[0, 0]
        img_out[img_ori == i] = smallest_idx
    return img_out


def draw_comparable_pic(img_ori, img_ref):
    img_matched = match(img_ori, img_ref)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Origin Image")
    plt.imshow(img_ori, cmap="gray")
    plt.subplot(2, 3, 2)
    plt.title("Referred Image")
    plt.imshow(img_ref, cmap="gray")
    plt.subplot(2, 3, 3)
    plt.title("Matched Image")
    plt.imshow(img_matched, cmap="gray")
    plt.subplot(2, 3, 4)
    plt.title("Origin Histogram")
    plt.hist(img_ori.ravel(), 256, [0, 256])
    plt.subplot(2, 3, 5)
    plt.title("Referred Histogram")
    plt.hist(img_ref.ravel(), 256, [0, 256])
    plt.subplot(2, 3, 6)
    plt.title("Matched Histogram")
    plt.hist(img_matched.ravel(), 256, [0, 256])
    plt.show()


if __name__ == '__main__':
    for img in citywall[2:]:
        draw_comparable_pic(img, citywall[1])
    for img in elain[2:]:
        draw_comparable_pic(img, elain[1])
    for img in lena[2:]:
        draw_comparable_pic(img, lena[1])
    for img in women[2:]:
        draw_comparable_pic(img, women[1])
