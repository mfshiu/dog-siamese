import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SiameseNetwork(nn.Module):
    def __init__(self, image_size=100):
        super(SiameseNetwork, self).__init__()
        dropout_rate = 0.2

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            # nn.ReflectionPad2d(1),
            # nn.Conv2d(8, 16, kernel_size=3),
            # nn.Dropout(dropout_rate),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(16),

            # nn.ReflectionPad2d(1),
            # nn.Conv2d(16, 16, kernel_size=3),
            # nn.Dropout(dropout_rate),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(16),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16 * image_size * image_size, 500),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def evaluate(self, x, y):
        out1, out2 = self(x, y)
        diss = nn.functional.pairwise_distance(out1, out2)
        similarity = 2 * (1 - math.fabs(self.sigmoid(diss))) * 100

        return similarity

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive