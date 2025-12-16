from __future__ import division
import copy
import torch
import torchvision.transforms.functional as TF
from torchvision import models
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from skimage.transform import radon
from PIL import Image
import numpy as np


WIDTH = 29
THETA = np.linspace(0.0, 360, WIDTH, endpoint=False)


class RA(torch.nn.Module):
    def __init__(self, group='C4'):
        super(RA, self).__init__()
        self.backbone = MnistCNN()
        self.group = group
        self.resize1 = Resize(87)  # to upsample
        self.resize2 = Resize(29)  # to downsample
        self.totensor = ToTensor()

    def forward(self, x):
        y = 0
        x = x.to('cpu')
        org_x = copy.deepcopy(x)

        if self.group == 'C1':
            G = [0]
        elif self.group == 'C4':
            G = [0, 90, 180, 270]
        elif self.group == 'C8':
            G = [0, 45, 90, 135, 180, 225, 270, 315]
        else:
            raise NameError

        if self.group != 'C1':
            for r in G:
                x = x.to('cpu')
                for i in range(x.shape[0]):
                    img = org_x[i][0].numpy()
                    img = Image.fromarray(img, mode='F')
                    np_x = self.totensor(self.resize2(self.resize1(img).rotate(r, Image.BILINEAR))).numpy()
                    sinogram = radon(np_x[0], theta=THETA)
                    sinogram = torch.FloatTensor(sinogram).reshape(1, 29, 29)
                    x[i] = sinogram

                x = x.to('cuda')
                y += self.backbone(x)

            y /= len(G)
            return y
        else:
            x = x.to('cpu')
            for i in range(x.shape[0]):
                np_x = org_x[i][0].numpy()
                sinogram = radon(np_x, theta=THETA)
                sinogram = torch.FloatTensor(sinogram).reshape(1, 29, 29)
                x[i] = sinogram

            x = x.to('cuda')
            y = self.backbone(x)
            return y


class MnistCNN(torch.nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.keep_prob = 0.5

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.fc1 = torch.nn.Linear(2048, 128, bias=True)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        self.fc2 = torch.nn.Linear(128, 10, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out
