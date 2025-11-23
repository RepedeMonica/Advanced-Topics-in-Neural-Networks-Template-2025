import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from torchvision.transforms import RandomErasing
import torch
from timm.data import resolve_data_config
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

cutmix = v2.CutMix(num_classes=100)
mixup = v2.MixUp(num_classes=100)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

DATASET_STATS = {
    "CIFAR10": {"mean": [0.4787, 0.4781, 0.4777], "std": [0.2671, 0.2678, 0.2674]},

    "CIFAR100": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},

    "MNIST": {"mean": [0.1307], "std": [0.3081]},  
    
    "OXFORDIIITPET": {"mean": [0.478, 0.4458, 0.3956], "std": [0.2627, 0.2572, 0.2653]}
}

def get_transforms(dataset_name, train=True, pretrained=False):
    mean = DATASET_STATS.get(dataset_name, {}).get("mean", [0.5, 0.5, 0.5])
    std = DATASET_STATS.get(dataset_name, {}).get("std", [0.25, 0.25, 0.25])

    if pretrained or dataset_name == "OXFORDIIITPET":
        crop_size = 224
    else:
        if dataset_name in ["CIFAR10", "CIFAR100"]:
            crop_size = 32
        elif dataset_name == "MNIST":
                crop_size = 28
        else:
            crop_size = 224
            
        
    if dataset_name in ["CIFAR10", "CIFAR100"]:
        if train:
            return T.Compose([
            T.RandomCrop(crop_size, padding=4),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),   
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
        return T.Compose([T.Resize((crop_size, crop_size)),T.ToTensor(), T.Normalize(mean=mean, std=std, inplace=True)])

    if dataset_name == "MNIST":
        if train:
            return T.Compose([T.Resize((crop_size, crop_size)), T.RandomRotation(10), T.ToTensor(), T.Normalize(mean=mean, std=std, inplace=True)])
        return T.Compose([T.Resize((crop_size, crop_size)), T.ToTensor(), T.Normalize(mean=mean, std=std, inplace=True)])


    if dataset_name == "OXFORDIIITPET":
        if train:
            return T.Compose([T.Resize((crop_size, crop_size)), T.RandomHorizontalFlip(), T.ToTensor(),
                              T.Normalize(mean=mean, std=std, inplace=True)])
        return T.Compose([T.Resize((crop_size, crop_size)), T.ToTensor(), T.Normalize(mean=mean, std=std, inplace=True)])

    raise ValueError("Dataset unknown")


def get_dataloaders(cfg):
    name = cfg['dataset']['name']
    batch = cfg['training']['batch_size']
    num_workers = cfg['training'].get('num_workers', 4)
    data_dir = cfg['dataset'].get('data_dir', './data')
    cutmix_mixup = cfg['dataset'].get('cutmix_mixup', False)

    if name == "CIFAR10":
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=get_transforms(name, True, cfg['model']))

        targets = np.array(train.targets)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(sss.split(np.arange(len(targets)), targets))
        from torch.utils.data import Subset
        train_ds = Subset(train, train_idx)
        val_ds   = Subset(train, val_idx)
        
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=get_transforms(name, False, cfg['model']['pretrained']))
        num_classes = 10

    elif name == "CIFAR100":
        train = datasets.CIFAR100(data_dir, train=True, download=True, transform=get_transforms(name, True, cfg['model']['pretrained']))
        
        targets = np.array(train.targets)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(sss.split(np.arange(len(targets)), targets))
        from torch.utils.data import Subset
        train_ds = Subset(train, train_idx)
        val_ds   = Subset(train, val_idx)
        
        test = datasets.CIFAR100(data_dir, train=False, download=True, transform=get_transforms(name, False, cfg['model']['pretrained']))
        num_classes = 100

    elif name == "MNIST":
        train = datasets.MNIST(data_dir, train=True, download=True, transform=get_transforms(name, True, cfg['model']['pretrained']))

        targets = np.array(train.targets)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(sss.split(np.arange(len(targets)), targets))
        from torch.utils.data import Subset
        train_ds = Subset(train, train_idx)
        val_ds   = Subset(train, val_idx)
        
        test = datasets.MNIST(data_dir, train=False, download=True, transform=get_transforms(name, False, cfg['model']['pretrained']))
        num_classes = 10

    elif name == "OXFORDIIITPET":
        train = datasets.OxfordIIITPet(data_dir, split='trainval', target_types='category', download=True, transform=get_transforms(name, True, cfg['model']['pretrained']))

        targets = np.array(train.targets)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(sss.split(np.arange(len(targets)), targets))
        from torch.utils.data import Subset
        train_ds = Subset(train, train_idx)
        val_ds   = Subset(train, val_idx)
        
        test = datasets.OxfordIIITPet(data_dir, split='test', target_types='category', download=True, transform=get_transforms(name, False, cfg['model']['pretrained']))
        num_classes = 37

    else:
        raise ValueError("Unknown dataset")

    if cutmix_mixup:
        train_loader = DataLoader(train_ds, batch_size=100, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, num_classes
