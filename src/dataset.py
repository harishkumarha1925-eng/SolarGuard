# src/dataset.py
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_transforms(size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((size,size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1,0.1,0.1,0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def make_dataloaders(train_dir, val_dir, batch_size=32, size=224, num_workers=4):
    train_ds = ImageFolder(train_dir, transform=get_transforms(size, True))
    val_ds = ImageFolder(val_dir, transform=get_transforms(size, False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes
