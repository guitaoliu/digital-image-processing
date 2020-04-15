import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    return cv2.convertScaleAbs(np.abs(img_ifft))


def wiener(img, h, k=0.01):
    img_move_fft = np.fft.fftshift(np.fft.fft2(img))
    move_function_fft = h(img_move_fft.shape)
    img_recovered = (
                            1 / move_function_fft * np.abs(move_function_fft) ** 2 / (
                            np.abs(move_function_fft) ** 2 + k)
                    ) * img_move_fft
    img_ifft = np.fft.ifft2(np.fft.ifftshift(img_recovered))
    return cv2.convertScaleAbs(np.abs(img_ifft))


def constrained_least_squares(img, h, gama=0.01):
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    p = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ])
    x_idx_border = img.shape[1] - p.shape[1]
    y_idx_border = img.shape[0] - p.shape[0]
    p_img_resize = cv2.copyMakeBorder(
        p, y_idx_border // 2 + 1, y_idx_border // 2, x_idx_border // 2 + 1, x_idx_border // 2,
        borderType=cv2.BORDER_CONSTANT, value=0)
    p_fft = np.fft.fftshift(np.fft.fft2(p_img_resize))
    move_function_fft = h(img_fft.shape)
    img_recovered_fft = (
                                1 / move_function_fft
                                * np.abs(move_function_fft) ** 2
                                / (np.abs(move_function_fft) ** 2 + gama * np.abs(p_fft) ** 2)
                        ) * img_fft
    img_ifft = np.fft.ifft2(np.fft.ifftshift(img_recovered_fft))
    return cv2.convertScaleAbs(np.abs(img_ifft))


if __name__ == '__main__':
    lena = cv2.imread('images/lena.bmp', 0)

    lena_moved = move(lena)
    lena_wiener_recover = wiener(lena_moved.copy(), move_h, k=0.0001)
    lena_squares_recover = constrained_least_squares(lena_moved.copy(), move_h, gama=0.005)

    lena_gaussian_noised = gaussian_noise(lena_moved, var=10)
    lena_gaussian_wiener_recover = wiener(lena_gaussian_noised.copy(), move_h, k=0.002)
    lena_gaussian_squares_recover = constrained_least_squares(lena_gaussian_noised.copy(), move_h, gama=0.005)

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.imshow(lena_moved, cmap='gray')
    plt.title('Moved Lena')
    plt.subplot(232)
    plt.imshow(lena_wiener_recover, cmap='gray')
    plt.title('Wiener Recovered Lena')
    plt.subplot(233)
    plt.imshow(lena_squares_recover, cmap='gray')
    plt.title('Constrained Squares Recovered Lena')
    plt.subplot(234)
    plt.imshow(lena_gaussian_noised, cmap='gray')
    plt.title('Gaussian Noised Moved Lena')
    plt.subplot(235)
    plt.imshow(lena_gaussian_wiener_recover, cmap='gray')
    plt.title('Wiener Recovered Noised Lena')
    plt.subplot(236)
    plt.imshow(lena_gaussian_squares_recover, cmap='gray')
    plt.title('Constrained Squares Recovered Noised Lena')
    plt.show()
