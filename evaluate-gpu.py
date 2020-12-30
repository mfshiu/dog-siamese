import math
import random
import numpy as np

import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from model import SiameseNetwork
import torchvision.datasets as dset
from config import Config
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img_size = 128
# threshold = 0.76
model_path = "./trained/DogSiamese.pkl"
use_gpu = False


class TestDataset(Dataset):
    def __init__(self, left_image, right_images):
        self.left_image = left_image
        self.right_images = right_images
        self.transform = transforms.Compose([transforms.Resize((100, 100)),
                                             transforms.ToTensor(),
                                             ])

    def __getitem__(self, idx):
        left_img = self.left_image
        right_img = self.right_images[idx]

        img0 = Image.open(left_img).convert("L")
        img1 = Image.open(right_img).convert("L")
        img0 = self.transform(img0)
        img1 = self.transform(img1)

        return img0, img1

    def __len__(self):
        return len(self.right_images)

class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __get_imgs(self):
        # we need to make sure approx 50% of images are in the same class
        if random.randint(0, 1):
            the_class = random.choice(self.imageFolderDataset.classes)
            img0_tuple, img1_tuple = random.sample([x for x in self.imageFolderDataset.imgs if the_class in x[0]], 2)
        else:
            class_1, class_2 = random.sample(self.imageFolderDataset.classes, 2)
            img0_tuple = random.choice([x for x in self.imageFolderDataset.imgs if class_1 in x[0]])
            img1_tuple = random.choice([x for x in self.imageFolderDataset.imgs if class_2 in x[0]])
        return img0_tuple, img1_tuple

    def __getitem__(self, index):
        img0_tuple, img1_tuple = self.__get_imgs()

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def calculate_far_frr(inferences, group_size, threshold):
    fa, fr = 0, 0
    dogs = len(inferences)
    group_count = int(dogs / group_size)
    for dog in range(dogs):
        dog_group = int(dog / group_size)
        for g in range(group_count):
            if g == dog_group:
                for i in range(group_size):
                    if not (inferences[dog][g * group_size + i] > threshold):
                        fr += 1
            else:
                for i in range(group_size):
                    if inferences[dog][g * group_size + i] > threshold:
                        fa += 1

    fa_base = dogs * dogs - dogs * group_size
    fr_base = dogs * group_size
    return fa / fa_base, fr / fr_base


def verify_dogs(test_model, left_dogs, right_dogs):
    inferences = []

    for left_dog in left_dogs:
        inf = []
        test_set = TestDataset(left_dog, right_dogs)
        test_dataloader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=0)
        for i, data in enumerate(test_dataloader):
            img0, img1 = data
            similarity = test_model.evaluate(img0.cuda(), img1.cuda())
            inf.append(similarity)
            print("\r%s | %s = %f" % (left_dog, right_dogs[i], similarity), end="")
        print()
        inferences.append(inf)

    return inferences

import sys

if __name__ == '__main__':
    model_path = Config.Evaluate.model_path
    inference_output_path = Config.Evaluate.inference_output_path
    eer_output_path = Config.Evaluate.eer_output_path
    dog_input_root = Config.Evaluate.dog_input_root
    dog_count = Config.Evaluate.dog_count
    group_size = Config.Evaluate.group_size

    print("Model path: %s" % (model_path,))
    print("Input path: %s" % (dog_input_root,))
    print("dog_count: %s" % (dog_count,))
    print("group_size: %s" % (group_size,))

    siam_test = SiameseNetwork().cuda()
    siam_test.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
    siam_test.eval()

    dog_paths = [x for x in os.walk(dog_input_root)]
    dog_paths.sort()
    img_paths = []
    for dog_path in dog_paths[1: dog_count + 1]:
        img_paths.extend(os.path.join(dog_path[0], x) for x in dog_path[2][:group_size])

    inferences = verify_dogs(siam_test, img_paths, img_paths)
    dogs = len(img_paths)
    group_count = int(dogs / group_size)
    header = [""]
    for g in range(group_count):
        for i in range(group_size):
            header.append("{}-{}".format(g+1, i+1))
    lines = ["\t".join(header) + "\n"]
    for dd in range(dogs):
        line = header[dd + 1] + "\t"
        d = 0
        for g in range(group_count):
            for i in range(group_size):
                line += "{}\t".format(inferences[dd][d])
                d += 1
        lines.append(line[:-1] + "\n")
    with open(inference_output_path, "w") as fp:
        fp.writelines(lines)

    lines = ["Threshold\tFAR\tFRR\n"]
    for i in range(101):
        far, frr = calculate_far_frr(inferences, group_size, threshold=i)
        lines.append("%d\t%f\t%f\n" % (i, far, frr))

    with open(eer_output_path, "w") as fp:
        fp.writelines(lines)


if __name__ == 'test__main__':
    folder_dataset = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)

    test_dataloader = DataLoader(siamese_dataset,
                                 shuffle=True,
                                 num_workers=8,
                                 batch_size=1)

    model_test = SiameseNetwork().cuda()
    model_test.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_test.eval()

    for i, data in enumerate(test_dataloader):
        img0, img1, label = data
        similarity = model_test.evaluate(img0.cuda(), img1.cuda())
    print()