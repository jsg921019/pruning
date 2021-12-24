import random
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset

class RotationTransform:

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def get_dataset(input_size=128):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rotate = RotationTransform(angles=[0, 180, 90, 270])

    train_dataset = ImageFolder("/opt/ml/data/train",
                        transforms.Compose([
                        transforms.Resize((input_size, input_size)),
                        transforms.RandomHorizontalFlip(),
                        rotate,
                        transforms.ToTensor(),
                        normalize,
                    ]))

    val_dataset = ImageFolder("/opt/ml/data/val",
                        transforms.Compose([
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    
    return train_dataset, val_dataset

def get_concat_dataset(input_size=128):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rotate = RotationTransform(angles=[0, 180, 90, 270])

    train_dataset = ImageFolder("/opt/ml/data/train",
                        transforms.Compose([
                        transforms.Resize((input_size, input_size)),
                        transforms.RandomHorizontalFlip(),
                        rotate,
                        transforms.ToTensor(),
                        normalize,
                    ]))

    valid_dataset = ImageFolder("/opt/ml/data/val",
                        transforms.Compose([
                        transforms.Resize((input_size, input_size)),
                        transforms.RandomHorizontalFlip(),
                        rotate,
                        transforms.ToTensor(),
                        normalize,
                    ]))
    
    return ConcatDataset([train_dataset, valid_dataset])


def get_weighted_sampler(concat=False):
    
    sample_freq = [1169, 4826, 1020, 2655, 4879, 1092]
    sample_weight = np.concatenate([[1/f]*f for f in sample_freq])
    sample_weight = torch.from_numpy(sample_weight)
    sampler = WeightedRandomSampler(sample_weight.type('torch.DoubleTensor'), len(sample_weight)//2)
    
    return sampler