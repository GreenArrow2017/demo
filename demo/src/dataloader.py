import torch
import torchvision
from torch.utils.data import DataLoader
import Augmentor
from torchvision import transforms


def CIFAR10_dataloader(train=True, batch_size=64, transform = None, shuffle=True):
    dataloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('data/',
                                                                        train=train,
                                                                        download=True,
                                                                        transform=transform),
                                             batch_size=batch_size,
                                             shuffle=shuffle)
    return dataloader


def MNIST_dataloader(train=True, batch_size=64, transform = None, shuffle=True):
    dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('data/',
                                                                        train=train,
                                                                        download=True,
                                                                        transform=transform),
                                             batch_size=batch_size,
                                             shuffle=shuffle)
    return dataloader

# 放缩，翻转再不同场景效果就完成不一样的了，放缩再CIFAR有用，再MNIST没用，我怀疑是MNIST的放缩使得图片有二义性，而CIFAR不会
# 只能一个个去试了
def img_normalize(t):
    c, x, y = t.shape
    t = t.view(c, -1)
    t = t - t.mean(dim=1, keepdim=True)
    t = t / t.std(dim=1, keepdim=True, unbiased=False)
    return t.view(c, x, y)
def augmented_pipeline():
    rotate_distort = Augmentor.Pipeline()
    rotate_distort.random_distortion(1.0, 4, 4, magnitude=1)
    rotate_distort.rotate(probability=1.0, max_left_rotation=10, max_right_rotation=10)

    cropsize = 25
    noncentered_crops = Augmentor.Pipeline()
    noncentered_crops.crop_by_size(1.0, cropsize, cropsize, centre=False)
    noncentered_crops.resize(1.0, 28, 28)

    return transforms.Compose([
                          noncentered_crops.torch_transform(),
                          rotate_distort.torch_transform(),
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                      ])
def nonaugmented_pipeline():
    centered_crops = Augmentor.Pipeline()
    cropsize = 25
    centered_crops.crop_by_size(1.0, cropsize, cropsize, centre=True)
    centered_crops.resize(1.0, 28, 28)

    return   transforms.Compose([
                          centered_crops.torch_transform(),
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                        ])

if __name__ == '__main__':
    MNIST_dataloader(True, 64)
    MNIST_dataloader(False, 1000)
