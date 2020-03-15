from struct import unpack, pack
import numpy as np


class Read8BitBMP:
    def __init__(self, file_path):
        file = open(file_path, "rb")
        self.bfType = unpack("h", file.read(2))[0]
        self.bfSize = unpack("i", file.read(4))[0]
        self.bfReserved1 = unpack("h", file.read(2))[0]
        self.bfReserved2 = unpack("h", file.read(2))[0]
        self.bfOffBits = unpack("i", file.read(4))[0]

        self.biSize = unpack("i", file.read(4))[0]
        self.biWidth = unpack("i", file.read(4))[0]
        self.biHeight = unpack("i", file.read(4))[0]
        self.biPlanes = unpack("h", file.read(2))[0]
        self.biBitCount = unpack("h", file.read(2))[0]
        self.biCompression = unpack("i", file.read(4))[0]
        self.biSizeImages = unpack("i", file.read(4))[0]
        self.biXPelsPerMeter = unpack("i", file.read(4))[0]
        self.biYPelsPerMeter = unpack("i", file.read(4))[0]
        self.biClrUsed = unpack("i", file.read(4))[0]
        self.biClrImportant = unpack("i", file.read(4))[0]

        if self.bfOffBits == 1078:
            self.color_map = np.zeros((256, 4), dtype=np.uint8)
            for i in range(256):
                self.color_map[i, 0] = unpack("B", file.read(1))[0]
                self.color_map[i, 1] = unpack("B", file.read(1))[0]
                self.color_map[i, 2] = unpack("B", file.read(1))[0]
                self.color_map[i, 3] = unpack("B", file.read(1))[0]

        self.file_width = int(
            4 *
            np.ceil(
                self.biWidth *
                self.biBitCount /
                32.))
        
        self.data_map = np.zeros(
            (self.file_width, self.biHeight), dtype=np.uint8)
            
        for i in range(self.biHeight):
            for j in range(self.file_width):
                self.data_map[i, j] = unpack("B", file.read(1))[0]

        file.close()

    def write(self, file_path):
        file = open(file_path, "wb")
        file.write(pack("h", self.bfType))
        file.write(pack("i", self.bfSize))
        file.write(pack("h", self.bfReserved1))
        file.write(pack("h", self.bfReserved2))
        file.write(pack("i", self.bfOffBits))

        file.write(pack("i", self.biSize))
        file.write(pack("i", self.biWidth))
        file.write(pack("i", self.biHeight))
        file.write(pack("h", self.biPlanes))
        file.write(pack("h", self.biBitCount))
        file.write(pack("i", self.biCompression))
        file.write(pack("i", self.biSizeImages))
        file.write(pack("i", self.biXPelsPerMeter))
        file.write(pack("i", self.biYPelsPerMeter))
        file.write(pack("i", self.biClrUsed))
        file.write(pack("i", self.biClrImportant))

        for e in self.color_map.flatten():
            file.write(pack("B", e))

        for e in self.data_map.flatten():
            file.write(pack("B", e))

        file.close()
