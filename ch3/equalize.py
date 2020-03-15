import cv2
import numpy as np
import matplotlib.pyplot as plt

from hist import citywall, lena, elain, women

def hist_equlize(imgs):
    n = 0
    for img in imgs[1:]:
        n += 1
        eq = cv2.equalizeHist(img)
        # cv2.imwrite('result/eq/' + imgs[0] + str(n) + '.bmp', np.hstack([img, eq]))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist(img.ravel(), 256, [0, 256])
        plt.title(imgs[0] + str(n) + "_ori")
        plt.subplot(1, 2, 2)
        plt.hist(eq.ravel(), 256, [0, 256])
        plt.title(imgs[0] + str(n) + "_eq")
        plt.show()

if __name__ == '__main__':
    hist_equlize(citywall)
    hist_equlize(lena)
    hist_equlize(elain)
    hist_equlize(women)
