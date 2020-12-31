# %matplotlib inline
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import SiameseNetwork, ContrastiveLoss
import sys
from config import Config


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


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

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def train():
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=Config.train_batch_size)

    global use_gpu
    if use_gpu:
        net = SiameseNetwork().cuda()
    else:
        net = SiameseNetwork().cpu()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    counter = []
    loss_history = []
    iteration_number= 0

    for epoch in range(0, Config.max_epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            if use_gpu:
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            else:
                img0, img1 , label = img0.cpu(), img1.cpu() , label.cpu()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    show_plot(counter,loss_history)


global use_gpu
use_gpu = False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        use_gpu = "gpu" == sys.argv[1]
    print('use_gpu：', use_gpu)

    folder_dataset = dset.ImageFolder(root=Config.train_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)

    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
