import numpy as np
import os

from Read8BitBMP import Read8BitBMP

bmp_file = Read8BitBMP(r"images\lena.bmp")
ori_data = bmp_file.data_map.copy()

for k in range(8):
    for i in range(bmp_file.biHeight):
        for j in range(bmp_file.file_width):
            bmp_file.data_map[i, j] = np.around(
                ori_data[i, j] / (2**(8 - k) - 1)) * (2**(8 - k) - 1)

    file_path = ''.join(["result\\q2\\image", str(k + 1), ".bmp"])
    bmp_file.write(file_path)
