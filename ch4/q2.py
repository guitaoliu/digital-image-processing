import numpy as np


def gaussian_func(x, y, sigma=1):
    return 1 / (2 * np.pi * sigma**2) * np.exp(- (x**2 + y**2) / (2 * sigma**2))


def mask_generate(k, mask='gaussian'):
    mask_out = np.zeros((k, k))
    if mask=='gaussian':
        for i in range(len(mask_out)):
            for j in range(len(mask_out[i])):
                x, y = j-len(mask_out[i])//2, i-len(mask_out)//2
                mask_out[i, j] = gaussian_func(x, y)

    return np.round(mask_out/mask_out[0, 0]).astype(np.int)


mask_3 = mask_generate(3)
mask_5 = mask_generate(5)
mask_7 = mask_generate(7)

if __name__ == "__main__":
    print(mask_3)
    print(mask_5)
    print(mask_7)
