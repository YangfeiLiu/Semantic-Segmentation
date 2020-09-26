import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000


def changeDetect(img1, img2):
    return np.bitwise_xor(img1, img2)


if __name__ == '__main__':
    root1 = '/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/tianzhibei/changedatection/labelA2016.tif'
    root2 = '/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/tianzhibei/changedatection/labelA2019.tif'
    img1 = np.array(Image.open(root1))
    img2 = np.array(Image.open(root2))
    img = changeDetect(img1, img2) * 255
    Image.fromarray(img).save('/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/tianzhibei/changedatection/a.tif')
    print(img)