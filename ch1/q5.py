import numpy as np
import os
import math

from Read8BitBMP import Read8BitBMP

def shear(bmp_file: Read8BitBMP, prop=1.5):
    bmp_file.biWidth = int(bmp_file.biWidth*prop)
    bmp_file.biSizeImages = bmp_file.biWidth * bmp_file.biHeight
    bmp_file.bfSize = bmp_file.biSizeImages + bmp_file.bfOffBits
    new_data = np.full(
        (bmp_file.biHeight, bmp_file.biWidth), 
        255, 
        dtype=np.uint8
        )
    
    for i in range(len(bmp_file.data_map)):
        for j in range(len(bmp_file.data_map[i])):
            new_data[i, int(j + i*(prop-1))] = bmp_file.data_map[i, j]
    
    bmp_file.data_map = new_data
    return bmp_file

def rotate(bmp_file: Read8BitBMP, theta=math.pi/6, inter="linear", zoom=True):
    ori_data = bmp_file.data_map

    size_rotated = np.dot(
        np.array([
            [np.abs(math.cos(theta)), np.abs(math.sin(theta))],
            [np.abs(math.sin(theta)), np.abs(math.cos(theta))],
        ]),
        np.array([bmp_file.biWidth, bmp_file.biHeight]).T
    )

    bmp_file.biWidth = int(4 * np.ceil(size_rotated[0]*bmp_file.biBitCount/32))
    bmp_file.biHeight = int(size_rotated[1])
    bmp_file.biSizeImages = bmp_file.biWidth * bmp_file.biHeight
    bmp_file.bfSize = bmp_file.biSizeImages  + bmp_file.bfOffBits
    new_data = np.full(
        (bmp_file.biHeight, bmp_file.biWidth), 
        255, 
        dtype=np.uint8
        )
    

    rotate_matrix = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])

    for i in range(len(ori_data)):
        for j in range(len(ori_data[i])):
            cen_i, cen_j = i-int(len(ori_data)/2), j-int(len(ori_data[i])/2)
            dst_j, dst_i = np.dot(
                rotate_matrix,
                np.array([cen_j, cen_i]).T,
            )
            dst_ori_i = min(bmp_file.biHeight-1, dst_i+int(bmp_file.biHeight/2))
            dst_ori_j = min(bmp_file.biWidth-1,  dst_j+int(bmp_file.biWidth/2))
            new_data[int(dst_ori_i), int(dst_ori_j)] = ori_data[i, j]

    if inter == "nearest":
        bmp_file.data_map = nearest_inter(ori_data, new_data, np.linalg.inv(rotate_matrix))
    elif inter == "linear":
        bmp_file.data_map = linear_inter(ori_data, new_data, np.linalg.inv(rotate_matrix))
    elif inter == "cubic":
        bmp_file.data_map = cubic_inter(ori_data, new_data, np.linalg.inv(rotate_matrix))
    else:
        print("Unsupported inter operation")

    if zoom:
        new_image = np.full(
            (2048, 2048),
            255,
            dtype=np.uint8,
            )

        i_prop = 2048/bmp_file.biHeight
        j_prop = 2048/bmp_file.biWidth

        for i in range(len(bmp_file.data_map)):
            for j in range(len(bmp_file.data_map[i])):
                new_image[int(i_prop*i), int(j_prop*j)] = bmp_file.data_map[i, j]

        bmp_file.biWidth = 2048
        bmp_file.biHeight = 2048
        bmp_file.biSizeImages = 2048 * 2048
        bmp_file.bfSize = bmp_file.biSizeImages + bmp_file.bfOffBits    

        bmp_file.data_map = zoom_operation(new_image, bmp_file.data_map, i_prop, j_prop, inter)

    return bmp_file

def zoom_operation(new_image, ori_data, i_prop, j_prop, inter):
    if inter=="nearest":
        expand_data = np.row_stack((ori_data, ori_data[-1, :]))
        expand_data = np.column_stack((expand_data, expand_data[:, -1]))
        for i in range(len(new_image)):
            for j in range(len(new_image[i])):
                x = int(np.around(i/i_prop))
                y = int(np.around(j/j_prop))
                new_image[i, j] = expand_data[x, y]
    elif inter=="linear":
        expand_data = np.row_stack((ori_data, ori_data[-1, :]))
        expand_data = np.column_stack((expand_data, expand_data[:, -1]))
        for i in range(len(new_image)):
            for j in range(len(new_image[i])):
                x1 = int(i/i_prop)
                y1 = int(j/j_prop)
                x2 = x1 + 1
                y2 = y1 + 1
                w = np.array([
                    (x2-i/i_prop)*(y2-j/j_prop),
                    (i/i_prop-x1)*(y2-j/j_prop),
                    (x2-i/i_prop)*(j/j_prop-y1),
                    (i/i_prop-x1)*(j/j_prop-y1)
                    ])
                xy = np.array([expand_data[x1, y1], expand_data[x2, y1], expand_data[x1, y2], expand_data[x2, y2]])
                new_image[i, j] = int(np.dot(w, xy))
    elif inter=="cubic":
        expand_data = np.vstack((ori_data, np.tile(ori_data[-1, :], (2, 1))))
        expand_data = np.vstack((expand_data[0, :], expand_data))
        expand_data = np.hstack((expand_data, np.tile(expand_data[:, [-1]], 2)))
        expand_data = np.hstack((expand_data[:, [0]], expand_data))
        for i in range(len(new_image)):
            for j in range(len(new_image[i])):
                x2, y2 = int(i/i_prop), int(j/j_prop)
                x1, x3, x4 = x2 - 1, x2 + 1, x2 + 2
                y1, y3, y4 = y2 - 1, y2 + 1, y2 + 2
                u, v = i/i_prop - x2, j/j_prop - y2
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
    else:
        print("Unsupported zoom inter operation")

    return new_image

def nearest_inter(ori_data, new_data, rotate_matrix):
    expand_data = np.row_stack((ori_data, ori_data[-1, :]))
    expand_data = np.column_stack((expand_data, expand_data[:, -1]))

    for i in range(len(new_data)):
        for j in range(len(new_data[i])):
            [dst_j, dst_i] = np.dot(
                rotate_matrix,
                np.array([
                    j - len(new_data[0])/2,
                    i - len(new_data)/2
                ]),
            )
            dst_ori_i = int(np.around(dst_i+len(expand_data)/2))
            dst_ori_j = int(np.around(dst_j+len(expand_data[0])/2))
            if dst_ori_i < len(expand_data) and dst_ori_j < len(expand_data[0]) and dst_ori_i >= 0 and dst_ori_j >= 0:
                new_data[i, j] = expand_data[dst_ori_i, dst_ori_j]

    return new_data

def linear_inter(ori_data, new_data, rotate_matrix):
    expand_data = np.row_stack((ori_data, ori_data[-1, :]))
    expand_data = np.column_stack((expand_data, expand_data[:, -1]))

    for i in range(len(new_data)):
        for j in range(len(new_data[i])):
            [dst_j, dst_i] = np.dot(
                rotate_matrix,
                np.array([
                    j - len(new_data[0])/2,
                    i - len(new_data)/2
                ]),
            )
            dst_i += len(expand_data)/2
            dst_j += len(expand_data[0])/2
            x1 = int(dst_i)
            y1 = int(dst_j)
            if x1 < len(expand_data)-1 and y1 < len(expand_data[0])-1 and x1 >= 0 and y1 >= 0:
                x2 = x1 + 1
                y2 = y1 + 1
                w = np.array([
                    (x2-dst_i)*(y2-dst_j),
                    (dst_i-x1)*(y2-dst_j),
                    (x2-dst_i)*(dst_j-y1),
                    (dst_i-x1)*(dst_j-y1),
                    ])
                xy = np.array([expand_data[x1, y1], expand_data[x2, y1], expand_data[x1, y2], expand_data[x2, y2]])
                new_data[i, j] = int(np.dot(w, xy))
    return new_data

def cubic_inter(ori_data, new_data, rotate_matrix):
    expand_data = np.vstack((ori_data, np.tile(ori_data[-1, :], (2, 1))))
    expand_data = np.vstack((expand_data[0, :], expand_data))
    expand_data = np.hstack((expand_data, np.tile(expand_data[:, [-1]], 2)))
    expand_data = np.hstack((expand_data[:, [0]], expand_data))

    for i in range(len(new_data)):
        for j in range(len(new_data[i])):
            [dst_j, dst_i] = np.dot(
                rotate_matrix,
                np.array([
                    j - len(new_data[0])/2,
                    i - len(new_data)/2
                ]),
            )
            dst_i += len(expand_data)/2
            dst_j += len(expand_data[0])/2
            x2, y2 = int(dst_i), int(dst_j)
            if x2 < len(expand_data)-2 and y2 < len(expand_data[0])-2 and x2 >= 0 and y2 >= 0:
                x1, x3, x4 = x2 - 1, x2 + 1, x2 + 2
                y1, y3, y4 = y2 - 1, y2 + 1, y2 + 2
                u, v = dst_i - x2, dst_j - y2
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
                new_data[i, j] = int(np.dot(np.dot(w1, xy),w2))

    return new_data

def main():
    path = "result/q5"
    if not os.path.exists(path):
        os.makedirs(path)

    lena = Read8BitBMP("images/lena.bmp")
    elain = Read8BitBMP("images/elain1.bmp")
    
    # Shear operation
    # lena_sheared = shear(lena)
    # lena_sheared.write(path + "/lena_sheared.bmp") 
    # elain_sheared = shear(elain)
    # elain_sheared.write(path + "/elain_sheared.bmp")

    methods = [
        "nearest",
        "linear",
        "cubic",
    ]

    for method in methods:
        lena = Read8BitBMP("images/lena.bmp")
        elain = Read8BitBMP("images/elain1.bmp")

        # Rotate operation
        lena_rotated = rotate(lena, theta=math.pi/6, inter=method)
        lena_rotated.write(path + "/lena_rotated_" + method + ".bmp")
        elain_rotated = rotate(elain, theta=math.pi/6, inter=method)
        elain_rotated.write(path + "/elain_rotated_" + method + ".bmp")
    
if __name__ == "__main__":
    main()
