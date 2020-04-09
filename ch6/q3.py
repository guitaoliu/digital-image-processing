import cv2
import numpy as np

from ch6.q1 import gaussian_noise


def move_h(shape, a=0.1, b=0.1, T=1):
    h = np.ones(shape, dtype=np.complex)
    for i in range(shape[0]):
        for j in range(shape[1]):
            u, v = i - shape[0] // 2, j - shape[1] // 2
            if u * a + v * b == 0:
                h[i, j] = 1
            else:
                h[i, j] = T * \
                          np.sin(np.pi * (u * a + v * b)) * \
                          np.exp(-np.pi * (u * a + v * b) * 1j) / \
                          (np.pi * (u * a + v * b))
    return h


def move(img):
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    img_fft_moved = img_fft * move_h(img_fft.shape)
    img_ifft = np.fft.ifft2(np.fft.ifftshift(img_fft_moved))
    return np.uint8(np.real(img_ifft))


def wiener(img, h, k=0.01):
    img_move_fft = np.fft.fftshift(np.fft.fft2(img))
    move_fft = move_h(img_move_fft.shape)
    # img_recovered = (np.abs(move_fft) ** 2 / (np.abs(move_fft) ** 2 + k)) * img_fft_moved / move_fft
    img_recovered = img_move_fft / move_fft
    img_ifft = np.fft.ifft2(np.fft.ifftshift(img_recovered))
    return np.uint8(np.real(img_ifft))


if __name__ == '__main__':
    lena = cv2.imread('images/lena.bmp', 0)
    lena_moved = move(lena)
    lena_gaussian_noised = gaussian_noise(lena_moved, var=10)

    lena_inverse_recover = wiener(lena_moved, move_h, k=0)
    lena_wiener_recover = wiener(lena_moved, move_h, k=0.01)

    lena_gaussian_wiener_recover = wiener(lena_gaussian_noised.copy(), move_h, k=0.01)

    cv2.imshow('lena blur', lena_moved)
    cv2.imshow('lena Gaussian Blur', lena_gaussian_noised)
    cv2.imshow('wiener recover', lena_wiener_recover)
    cv2.waitKey(0)
