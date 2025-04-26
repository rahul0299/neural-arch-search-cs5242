import torch
import torch.nn.functional as F
import random

import utils

class FastTensorAugment:
    def __init__(self, crop_size=32, padding=4, hflip_prob=0.5,
                 mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761)):
        self.crop_size = crop_size
        self.padding = padding
        self.hflip_prob = hflip_prob
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img):
        # Reflection padding
        img = F.pad(img, pad=[self.padding] * 4, mode='reflect')
        top = random.randint(0, img.size(1) - self.crop_size)
        left = random.randint(0, img.size(2) - self.crop_size)
        # Random crop based on probability
        img = img[:, top:top + self.crop_size, left:left + self.crop_size]

        # Flipping image based on probability
        if random.random() < self.hflip_prob:
            img = torch.flip(img, dims=[2])


        mean = self.mean.to(img.device)
        std = self.std.to(img.device)
        img = (img - mean) / std

        return img


def get_fast_transform(dataset_name):
    mean, std = get_dataset_stats(dataset_name)

    # Recommended crop/pad per dataset
    if dataset_name.lower() == "mnist":
        crop_size = 28
        padding = 2
    else: #CIFAR,CIFAR100
        crop_size = 32
        padding = 4

    return FastTensorAugment(
        crop_size=crop_size,
        padding=padding,
        mean=mean,
        std=std
    )

def normalize_eval_tensor(tensor, dataset_name, device=None):
    if device is None:
        device = utils.get_device_available()

    mean, std = get_dataset_stats(dataset_name)
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(device)
    if tensor.max() > 1:
        tensor = tensor.float() / 255.0
    return (tensor - mean) / std

def get_dataset_stats(dataset_name):
    dataset_name = dataset_name.lower()

    # Pre-computed mean and std of each dataset's images
    if dataset_name == "cifar100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2761)
    elif dataset_name == "cifar":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == "mnist":
        mean = (0.1307,)
        std = (0.3081,)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported in get_dataset_stats().")

    return mean, std