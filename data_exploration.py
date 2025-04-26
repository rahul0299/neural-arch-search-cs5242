import torch
import utils
import matplotlib.pyplot as plt
from collections import Counter
import random
import math

def explore_dataset(dataset="cifar", data_path="../../data/"):

    if dataset == "cifar":
        data_path = utils.check_cifar_dataset_exists(data_path)
    elif dataset == "mnist":
       data_path = utils.check_mnist_dataset_exists(data_path)
    elif dataset == "cifar100":
        data_path = utils.check_cifar100_dataset_exists(data_path)
    else:
        raise ValueError(f"Dataset '{dataset}' not supported in explore_dataset().")

    train_x, train_y = (torch.load(data_path + f'{dataset}/train_data.pt', weights_only=True), torch.load(data_path + f'{dataset}/train_label.pt', weights_only=True))

    test_x, test_y = (torch.load(data_path + f'{dataset}/test_data.pt', weights_only=True), torch.load(data_path + f'{dataset}/test_label.pt', weights_only=True))

    print("Number of Training Samples:", len(train_x))
    print("Number of Test Samples:", len(test_x))

    num_channels = 1

    if len(train_x.size())==4:
        num_channels=train_x.size(1)

    img_size = train_x.size(-1)

    num_classes = train_y.unique().size(0)

    print(f"Image size: {img_size}")
    print(f"Number of channels: {num_channels}")
    print(f"Number of classes:  {num_classes}")

    counts = Counter(train_y.tolist())
    plt.bar(counts.keys(), counts.values())
    plt.title(f"{dataset} Train Set Class Distribution")
    plt.xlabel("Class number")
    plt.ylabel("Count")

    counts = Counter(test_y.tolist())
    plt.bar(counts.keys(), counts.values())
    plt.title(f"{dataset} Test Set Class Distribution")

    plt.legend(["Train", "Test"])

    # Show 1 image per class
    # Sample either all classes or 10 random ones
    preview_samples = torch.arange(num_classes) if num_classes <= 10 else torch.randint(0, num_classes, (10,))
    num_preview = len(preview_samples)

    cols = math.ceil(num_preview / 2)
    rows = 2

    plt.figure(figsize=(cols * 2.5, rows * 2.5))

    for i, class_id in enumerate(preview_samples):
        class_id = class_id.item()

        indices = (train_y == class_id).nonzero(as_tuple=True)[0]
        rand_idx = random.choice(indices).item()
        image = train_x[rand_idx]

        plt.subplot(rows, cols, i + 1)
        if num_channels == 1:
            plt.imshow(image.squeeze(0), cmap="gray")
        else:
            plt.imshow(image.permute(1, 2, 0))
        plt.title(f"Class {class_id}")
        plt.axis("off")

    plt.suptitle("Random Image per Class (up to 10 shown)")
    plt.tight_layout()
    plt.show()


    plt.tight_layout()
    plt.show()
