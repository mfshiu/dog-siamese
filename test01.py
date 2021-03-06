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

size = (100, 100)
img0.thumbnail(size)
img1.thumbnail(size)

img0.save("/Users/xumingfang/Documents/Temp/060-01a.png")
img1.save("/Users/xumingfang/Documents/Temp/060-02a.png")

img0 = PIL.ImageOps.equalize(img0)
img1 = PIL.ImageOps.equalize(img1)

img0.save("/Users/xumingfang/Documents/Temp/060-01b.png")
img1.save("/Users/xumingfang/Documents/Temp/060-02b.png")
