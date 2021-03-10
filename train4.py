import os
import random
import sys
##
import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from torch import optim
from torch.utils.data import DataLoader, Dataset
import cv2

from config import Config
from model import SiameseNetwork, ContrastiveLoss
# from model2 import SiameseNetwork, ContrastiveLoss

global use_gpu
use_gpu = False

image_size = 100

trained_dir = "trained"
if not os.path.exists(trained_dir):
    os.makedirs(trained_dir)


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

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

    def __get_imgs2(self, idx):
        idx %= len(self.imageFolderDataset.classes)
        # we need to make sure approx 50% of images are in the same class
        if random.randint(1, 12) <= 4:
            # choose the same dogs
            the_class = self.imageFolderDataset.classes[idx]
            img0_tuple, img1_tuple = random.sample([x for x in self.imageFolderDataset.imgs if the_class in x[0]], 2)
        else:
            # choose the different dogs
            class_1 = self.imageFolderDataset.classes[idx]
            class_2 = random.choice(self.imageFolderDataset.classes)
            while class_1 == class_2:
                class_2 = random.choice(self.imageFolderDataset.classes)
            img0_tuple = random.choice([x for x in self.imageFolderDataset.imgs if class_1 in x[0]])
            img1_tuple = random.choice([x for x in self.imageFolderDataset.imgs if class_2 in x[0]])
        return img0_tuple, img1_tuple

    def __get_one_of_five(self, img, index):
        # return random.choice(transforms.Compose([
        #     transforms.TenCrop(image_size)
        # ])(img))
        return transforms.Compose([
            transforms.FiveCrop(image_size)
        ])(img)[index]

    def __getitem__(self, index):
        print("\rDataset get image index %s" % (index,), end='')
        img0_tuple, img1_tuple = self.__get_imgs2(index)

        img0 = Image.open(img0_tuple[0]).resize((image_size * 4, image_size * 4)).crop(50, 50, 350, 350)
        img1 = Image.open(img1_tuple[0]).resize((image_size * 4, image_size * 4)).crop(50, 50, 350, 350)
        piece_index = random.randint(0, 4)
        img0 = self.__get_one_of_five(img0, piece_index).convert("L")
        img1 = self.__get_one_of_five(img1, piece_index).convert("L")
        img0 = PIL.ImageOps.equalize(img0)
        img1 = PIL.ImageOps.equalize(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs) * 100


def train(model_path):
    print("Training dir: ", Config.train_dir)
    folder_dataset = dset.ImageFolder(root=Config.train_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([
                                                transforms.RandomAffine(degrees=10, translate=(0, 0.05),
                                                                        scale=(0.95, 1.05), shear=(5, 5)),
                                                transforms.ToTensor(),
                                            ]))

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=32,
                                  batch_size=Config.train_batch_size)

    global use_gpu
    if use_gpu:
        net = SiameseNetwork(image_size).cuda()
    else:
        net = SiameseNetwork(image_size).cpu()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, Config.max_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            if use_gpu:
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            else:
                img0, img1, label = img0.cpu(), img1.cpu(), label.cpu()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("\nEpoch number {}/{}\n Current loss {}\n".format(epoch, Config.max_epochs, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    if model_path:
        torch.save(net.state_dict(), model_path)

    # show_plot(counter,loss_history)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        use_gpu = "gpu" == sys.argv[1]
    print('use_gpu：', use_gpu)

    model_path = os.path.join(trained_dir, "DogSiamese.pkl")
    train(model_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
