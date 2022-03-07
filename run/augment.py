from torchvision import transforms
import numpy as np
import Augmentor


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

    return transforms.Compose([noncentered_crops.torch_transform(), rotate_distort.torch_transform(), transforms.ToTensor(), transforms.Lambda(img_normalize)])


def nonaugmented_pipeline():
    centered_crops = Augmentor.Pipeline()
    cropsize = 25
    centered_crops.crop_by_size(1.0, cropsize, cropsize, centre=True)
    centered_crops.resize(1.0, 28, 28)

    return transforms.Compose([centered_crops.torch_transform(), transforms.ToTensor(), transforms.Lambda(img_normalize)])


# 扭曲+旋转+裁剪
def distort_roteate_crop_transform():
    tranform = augmented_pipeline()
    return tranform


# 只有裁剪
# 裁剪在mnist和cifar中的效果都不是很好，基本是没有效果的了
def crop_transform():
    transform = nonaugmented_pipeline()
    return transform


# 只有旋转
# 效果不明显，cifar有一点
def rotation_transform():
    transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    return transform


# 映射+旋转
def aff_rotate_transform():
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomRotation((-10, 10)),  # 将图片随机旋转（-10,10）度
        transforms.ToTensor(),  # 将PIL图片或者numpy.ndarray转成Tensor类型
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    return transform


# 加了翻转
'''
这个垂直翻转在cifar很有用，但是在mnist不行，并且水平翻转在cifar起副作用
mnist感觉都差不多，主要是loss和scheduler对mnist的作用大
'''


def flip_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform
