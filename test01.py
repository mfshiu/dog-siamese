import PIL.ImageOps
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import math

img0_tuple = ("/Users/xumingfang/Documents/Temp/060-01.png",)
img1_tuple = ("/Users/xumingfang/Documents/Temp/060-02.png",)

img0 = Image.open(img0_tuple[0])
img1 = Image.open(img1_tuple[0])

img0 = img0.convert("L")
img1 = img1.convert("L")

img0a = PIL.ImageOps.equalize(img0)
img0a = ImageEnhance.Sharpness(img0a).enhance(10.0)
img1a = PIL.ImageOps.equalize(img1)
img1a = ImageEnhance.Sharpness(img1a).enhance(10.0)

img0a.save("/Users/xumingfang/Documents/Temp/060-01a.png")
img1a.save("/Users/xumingfang/Documents/Temp/060-02a.png")

img0b = PIL.ImageOps.equalize(img0)
img0b = ImageEnhance.Sharpness(img0b).enhance(100.0)
img1b = PIL.ImageOps.equalize(img1)
img1b = ImageEnhance.Sharpness(img1b).enhance(100.0)

img0b.save("/Users/xumingfang/Documents/Temp/060-01b.png")
img1b.save("/Users/xumingfang/Documents/Temp/060-02b.png")

# w = 120
# img0.thumbnail((w, w))
# img1.thumbnail((w, w))
#
# d = w - 100
# crop_size = (d, d, w-d, w-d)
# img0 = img0.crop(crop_size)
# img1 = img1.crop(crop_size)

# img0 = PIL.ImageOps.equalize(img0)
# img1 = PIL.ImageOps.equalize(img1)
#
# img0.save("/Users/xumingfang/Documents/Temp/060-01b.png")
# img1.save("/Users/xumingfang/Documents/Temp/060-02b.png")
