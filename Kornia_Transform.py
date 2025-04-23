import torch
import kornia.augmentation as K
import utils

def get_dataset_stats(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2761)
    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dataset_name == "mnist":
        mean = (0.1307,)
        std = (0.3081,)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported in get_dataset_stats().")

    return mean, std

def normalize_eval_tensor(tensor, dataset_name, device=None):

    if device is None:
        device = utils.get_device_available()

    mean, std = get_dataset_stats(dataset_name)
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(device)
    if tensor.max() > 1:
        tensor = tensor.float() / 255.0
    return (tensor - mean) / std

# def normalize_eval_tensor(tensor, dataset_name, device=None):
#     if device is None:
#         device = utils.get_device_available()
#
#     mean, std = get_dataset_stats(dataset_name)
#
#     if tensor.max() > 1:
#         tensor = tensor.float() / 255.0
#
#     normalize = K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)).to(device)
#     return normalize(tensor)

def get_kornia_transform(dataset_name):
    dataset_name = dataset_name.lower()
    mean, std = get_dataset_stats(dataset_name)

    if dataset_name == "imagenet":
        crop_size = 224
        padding = 32
    elif dataset_name == "mnist":
        crop_size = 28
        padding = 2
    else:
        crop_size = 32
        padding = 4

    transform = torch.nn.Sequential(
        K.RandomCrop((crop_size, crop_size), padding=padding, padding_mode='reflect'),
        K.RandomHorizontalFlip(p=0.5),
        K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    )

    return transform


