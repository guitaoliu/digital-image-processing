import matplotlib.pyplot as plt
import numpy as np


def butterworth(p, q, n=2, radius=25):
    low_pass_filter = np.zeros((p, q), complex)
    high_pass_filter = np.zeros((p, q), complex)
    d = np.zeros((p, q), complex)
    for u in range(p):
        for v in range(q):
            d[u, v] = np.sqrt((u - p / 2) ** 2 + (v - q / 2) ** 2)
            low_pass_filter[u, v] = 1 / (1 + (d[u, v] / radius) ** (2 * n))
            if d[u, v] == complex(0, 0):
                continue
            high_pass_filter[u, v] = 1 / (1 + (radius / d[u, v]) ** (2 * n))

    return low_pass_filter, high_pass_filter


def gaussian(p, q, radius=25):
    low_pass_filter = np.zeros((p, q), complex)
    high_pass_filter = np.zeros((p, q), complex)
    d = np.zeros((p, q), complex)
    for u in range(p):
        for v in range(q):
            d[u, v] = np.sqrt((u - p / 2) ** 2 + (v - q / 2) ** 2)
            low_pass_filter[u, v] = np.exp(-d[u, v] ** 2 / (2 * radius ** 2))
            high_pass_filter[u, v] = 1 - low_pass_filter[u, v]

    return low_pass_filter, high_pass_filter


def laplace(p, q):
    laplace_filter = np.zeros((p, q), complex)
    d = np.zeros((p, q), complex)
    for u in range(p):
        for v in range(q):
            d[u, v] = np.sqrt((u - p / 2) ** 2 + (v - q / 2) ** 2)
            laplace_filter[u, v] = - 4 * np.pi ** 2 * d[u, v] ** 2
    return laplace_filter


def unsharp(p, q, radius=20, k=1):
    _, GHPF = gaussian(p, q, radius)
    return 1 + k * GHPF


def show_filters(filters):
    fig = plt.figure()
    for n, filter in enumerate(filters):
        ax = fig.add_subplot(1, len(filters), n + 1, projection='3d')
        x, y = np.meshgrid(range(filter.shape[0]), range(filter.shape[1]))
        ax.plot_surface(x, y, np.real(filter), cmap=plt.get_cmap('rainbow'))
        # plt.title(f'radius = {25*(n+1)}')
    plt.show()


if __name__ == '__main__':
    p, q = 512, 512
    # GLPFs, GHPFs = [], []
    # BLPFs, BHPFs = [], []
    # for radius in range(25, 100, 25):
    #     GLPF, GHPF = gaussian(p, q, radius)
    #     GLPFs.append(GLPF)
    #     GHPFs.append(GHPF)
    #     BLPF, BHPF = butterworth(p, q, radius=radius)
    #     BLPFs.append(BLPF)
    #     BHPFs.append(BHPF)
    # show_filters(GLPFs)
    # show_filters(GHPFs)
    # show_filters(BLPFs)
    # show_filters(BHPFs)

    show_filters([laplace(p, q)])
