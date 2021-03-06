import PIL.ImageOps
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import math

img0_tuple = ("data/ct0202a/045/045-03.png",)
img1_tuple = ("data/ct0202a/072/072-04.png",)

img0 = Image.open(img0_tuple[0])
img1 = Image.open(img1_tuple[0])

img0 = img0.convert("L")
img1 = img1.convert("L")

img0.save("data/output/045-01.png")
img1.save("data/output/045-03.png")

img0 = PIL.ImageOps.equalize(img0)
img1 = PIL.ImageOps.equalize(img1)

img0.save("data/output/045-01a.png")
img1.save("data/output/045-03a.png")
