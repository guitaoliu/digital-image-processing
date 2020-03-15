import numpy as np
import time
import os

from Read8BitBMP import Read8BitBMP

# 最近邻插值
def nearest_inter(new_image, ori_data):
    expand_data = np.row_stack((ori_data, ori_data[-1, :]))
    expand_data = np.column_stack((expand_data, expand_data[:, -1]))
  
    start = time.time()
    for i in range(len(new_image)):
        for j in range(len(new_image[i])):
            x = int(np.around(i/4.))
            y = int(np.around(j/4.))
            new_image[i, j] = expand_data[x, y]
    end = time.time()
    print("The process time of nearest interpolaion algorithm is {}".format(end - start))
    return new_image

# 双线性
def linear_inter(new_image, ori_data):
    start = time.time()
    expand_data = np.row_stack((ori_data, ori_data[-1, :]))
    expand_data = np.column_stack((expand_data, expand_data[:, -1]))

    for i in range(len(new_image)):
        for j in range(len(new_image[i])):
            x1 = int(i/4)
            y1 = int(j/4)
            x2 = x1 + 1
            y2 = y1 + 1
            w = np.array([(x2-i/4.)*(y2-j/4.),(i/4.-x1)*(y2-j/4.),(x2-i/4.)*(j/4.-y1),(i/4.-x1)*(j/4.-y1)])
            xy = np.array([expand_data[x1, y1], expand_data[x2, y1], expand_data[x1, y2], expand_data[x2, y2]])
            new_image[i, j] = int(np.dot(w, xy))
    end = time.time()
    print("The process time of linear interpolaion algorithm is {}s".format(end - start))
    return new_image

# 三次
def cubic_inter(new_image, ori_data):
    start = time.time()
    expand_data = np.vstack((ori_data, np.tile(ori_data[-1, :], (2, 1))))
    expand_data = np.vstack((expand_data[0, :], expand_data))
    expand_data = np.hstack((expand_data, np.tile(expand_data[:, [-1]], 2)))
    expand_data = np.hstack((expand_data[:, [0]], expand_data))
    for i in range(len(new_image)):
        for j in range(len(new_image[i])):
            x2, y2 = int(i/4), int(j/4)
            x1, x3, x4 = x2 - 1, x2 + 1, x2 + 2
            y1, y3, y4 = y2 - 1, y2 + 1, y2 + 2
            u, v = i/4. - x2, j/4. - y2
            w1 = np.array([
                4 - 8*np.abs(u+1)    + 5*np.abs(u+1)**2 - np.abs(u+1)**3,
                1 - 2*np.abs(u)**2   + np.abs(u)**3,
                1 - 2*np.abs(u-1)**2 + np.abs(u-1)**3,
                4 - 8*np.abs(u-2)    + 5*np.abs(u-2)**2 - np.abs(u-2)**3,
            ])
            w2 = np.array([
                4 - 8*np.abs(v+1)    + 5*np.abs(v+1)**2 - np.abs(v+1)**3,
                1 - 2*np.abs(v)**2   + np.abs(v)**3,
                1 - 2*np.abs(v-1)**2 + np.abs(v-1)**3,
                4 - 8*np.abs(v-2)    + 5*np.abs(v-2)**2 - np.abs(v-2)**3,                
            ])
            xy = np.array([
                [expand_data[x1, y1], expand_data[x1, y2], expand_data[x1, y3], expand_data[x1, y4]],
                [expand_data[x2, y1], expand_data[x2, y2], expand_data[x2, y3], expand_data[x2, y4]],
                [expand_data[x3, y1], expand_data[x3, y2], expand_data[x3, y3], expand_data[x3, y4]],
                [expand_data[x4, y1], expand_data[x4, y2], expand_data[x4, y3], expand_data[x4, y4]],
            ])
            new_image[i, j] = int(np.dot(np.dot(w1, xy),w2))
    end = time.time()
    print("The process time of cubic interpolaion algorithm is {}s".format(end - start))
    return new_image

def main():
    bmp_file = Read8BitBMP("images/lena.bmp")
    ori_data = bmp_file.data_map.copy()

    new_image = np.zeros((2048, 2048), dtype=np.uint8)
    for i in range(len(ori_data)):
        for j in range(len(ori_data[i])):
            new_image[4 * i, 4 * j] = ori_data[i, j]

    bmp_file.biWidth = 2048
    bmp_file.biHeight = 2048
    bmp_file.biSizeImages = 2048 * 2048
    bmp_file.bfSize = bmp_file.biSizeImages + bmp_file.bfOffBits

    path = "result/q4"
    if not os.path.exists(path):
        os.makedirs(path)

    bmp_file.data_map = nearest_inter(new_image, ori_data)
    bmp_file.write(path + "/nearest_inter.bmp")

    bmp_file.data_map = linear_inter(new_image, ori_data)
    bmp_file.write(path + "/liner_inter.bmp")

    bmp_file.data_map = cubic_inter(new_image, ori_data)
    bmp_file.write(path + "/cubic_inter.bmp")

if __name__ == "__main__":
    main()