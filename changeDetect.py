import numpy as np
from PIL import Image


def changeDetect(img1, img2):
    return np.bitwise_xor(img1, img2)


if __name__ == '__main__':
    img1 = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.uint8)
    img2 = np.array([[2,2,3,3],[5,6,7,7]], dtype=np.uint8)
    img = changeDetect(img1, img2)
    print(img)