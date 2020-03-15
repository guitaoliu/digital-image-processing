import numpy as np

from Read8BitBMP import Read8BitBMP

bmp_file = Read8BitBMP(r"images\lena.bmp")
ori_data = bmp_file.data_map.copy()

lena_mean = np.mean(ori_data)
lena_var = np.var(ori_data)

print("lena 图像的均值为：{:.2f}".format(lena_mean))
print("lena 图像的方差为：{:.2f}".format(lena_var))