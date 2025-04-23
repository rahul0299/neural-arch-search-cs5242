from torch.utils.data import Dataset

class AugmentedTensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = img.float() / 255.0 if img.max() > 1 else img
        if self.transform:
            img = self.transform(img)
        return img, label
