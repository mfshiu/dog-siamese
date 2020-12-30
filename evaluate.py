import math
import random

import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from model import SiameseNetwork
import torchvision.datasets as dset
from config import Config


output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img_size = 128
# threshold = 0.76
model_path = "./trained/DogSiamese.pkl"
use_gpu = False


def sigmoid(x):
    return 1/(1+math.exp(-x))


transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.ToTensor()
                                ])

if __name__ == '__main__':
    folder_dataset = dset.ImageFolder(root=Config.testing_dir)
    the_class = random.choice(folder_dataset.classes)
    img0_tuple, img1_tuple = random.sample([x for x in folder_dataset.imgs if the_class in x[0]], 2)
    img0 = Image.open(img0_tuple[0]).convert("L")
    img1 = Image.open(img1_tuple[0]).convert("L")
    img0 = transform(img0)
    img1 = transform(img1)

    model_test = SiameseNetwork().cpu()
    model_test.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_test.eval()
    diss = model_test.evaluate(img0.cpu(), img1.cpu())
    similarity = 2 * (1 - math.fabs(sigmoid(diss))) * 100
    print("similarity: ", similarity)
