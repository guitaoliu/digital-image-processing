import cv2
import numpy as np
import matplotlib.pyplot as plt

from hist import elain, women


def elain_split():
    elain_ori = elain[1]
    plt.figure()
    plt.hist(elain_ori.ravel(), 256, [0, 256])
    plt.show()
    elain_hist = cv2.calcHist([elain_ori], [0], None, [256], [0, 256])
    min_indice = np.where(np.isclose(elain_hist, np.min(elain_hist[170:230])))[0][0]
    max_indice = np.where(np.isclose(elain_hist, np.max(elain_hist[100:150])))[0][0]
    print(min_indice)
    elain_ori[elain_ori >= max_indice] = 255
    elain_ori[elain_ori < max_indice] = 0
    return elain_ori
    # plt.figure()
    # plt.imshow(elain_ori, cmap="gray")
    # plt.show()

def women_split():
    women_ori = women[1]
    plt.figure()
    plt.hist(women_ori.ravel(), 256, [0, 256])
    plt.show()
    women_hist = cv2.calcHist([women_ori], [0], None, [256], [0, 256])
    min_indice = np.where(np.isclose(women_hist, np.min(women_hist[170:230])))[0][0]
    max_indice = np.where(np.isclose(women_hist, np.max(women_hist[100:150])))[0][0]
    print(min_indice)
    women_ori[women_ori >= max_indice] = 255
    women_ori[women_ori < max_indice] = 0
    return women_ori
    # plt.figure()
    # plt.imshow(women_ori, cmap="gray")
    # plt.show()

if __name__ == '__main__':
    elain_ori = elain_split()
    women_ori = women_split()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(elain_ori, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(women_ori, cmap='gray')
    plt.show()
