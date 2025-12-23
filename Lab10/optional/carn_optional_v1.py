#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install timed-decorator


# In[2]:


#pip install wandb tensorboard


# In[3]:


from multiprocessing import freeze_support
from timed_decorator.simple_timed import timed

import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np

from torchvision.utils import save_image
import os

import wandb


# In[4]:


device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")


# In[5]:


def get_cifar10_images(data_path: str, train: bool):
    initial_transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    cifar_10_images = CIFAR10(root=data_path, train=train, transform=initial_transforms, download=True)
    return [image for image, label in cifar_10_images]


# In[6]:


class CustomDataset(Dataset):
    def __init__(self, data_path: str = './data', train: bool = True, cache: bool = True):
        self.images = get_cifar10_images(data_path, train)
        self.cache = cache
        self.transforms = v2.Compose([
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
        ])
        if cache:
            self.labels = [self.transforms(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.cache:
            return self.images[i], self.labels[i]
        return self.images[i], self.transforms(self.images[i])


# In[7]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(3*32*32, 1*28*28)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(x.size(0), 1, 28, 28)
        return x


# In[8]:


@timed(return_time=True, use_seconds=True, stdout=False)
def transform_dataset_with_transforms(dataset: TensorDataset):
    transforms = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])
    for image in dataset.tensors[0]:
        transforms(image)

@timed(return_time=True, use_seconds=True, stdout=False)
@torch.no_grad()
def transform_dataset_with_model(dataset: TensorDataset, model: nn.Module, batch_size: int, device: torch.device):
    model.eval()  
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                           shuffle=False,
                           num_workers=(4 if device.type=="cuda" else 0),
                           pin_memory=(device.type=="cuda"))  # TODO: Complete the other parameters
    for (images,) in dataloader:
        images = images.to(device, non_blocking=True)
        model(images)  # TODO: uncomment this
        #pass

    if device.type == "cuda":
        torch.cuda.synchronize()


# In[9]:


def training_model(model, train_loader, optim, criterion, device):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)

        loss.backward()
        optim.step()

        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_loader.dataset)
    return train_loss


# In[10]:


def val_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item() * x.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss


# In[11]:


def train_model(
    device: torch.device,
    data_path: str = "./data",
    out_path: str = "./weights.pt",
    batch_size: int = 256,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 10,
    min_delta: float = 1e-4,
):

    wandb.init(project="CARN-optional")

    full = CustomDataset(data_path=data_path, train=True, cache=True)
    n = len(full)
    n_val = max(5000, n // 10)
    train_ds, val_ds = random_split(full, [n - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=(0 if device.type=="cpu" else 4), pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=(0 if device.type=="cpu" else 4), pin_memory=(device.type == "cuda"))

    model = Model().to(device)
    criterion = nn.MSELoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    bad = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = training_model(model, train_loader, optim, criterion, device)
        val_loss = val_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | train={train_loss:.6f} | val={val_loss:.6f}")
        wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss})

        if best_val - val_loss > min_delta:
            best_val = val_loss
            bad = 0
            torch.save({"model_state": model.state_dict()}, out_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping after epoch {epoch}. Best val={best_val:.6f}")
                break

    model.load_state_dict(torch.load(out_path, map_location=device)["model_state"])
    wandb.finish()
    return model


# In[12]:


def test_inference_time(model: nn.Module, device=torch.device('cpu'), batch_size: int = 128):
    test_dataset = CustomDataset(train=False, cache=False)
    test_dataset = torch.stack(test_dataset.images)
    test_dataset = TensorDataset(test_dataset)

    #batch_size = 100  # TODO: add the other parameters (device, ...)

    _, t1 = transform_dataset_with_transforms(test_dataset)

    model = model.to(device)
    _, t2 = transform_dataset_with_model(test_dataset, model, batch_size, device)

    print(f"Sequential transforming each image took: {t1}s on CPU. \n"
          f"Using a model with batch_size: {batch_size} took {t2}s on {device.type}. \n")


# In[13]:


def main():
    model = train_model(device=device)
    test_inference_time(model, device, batch_size=128)


if __name__ == '__main__':
    freeze_support()
    main()


# In[14]:


def save_comparison_images(device, out_dir="latex_images", num_images=5):
    model = Model()
    model.load_state_dict(torch.load("./weights.pt", map_location=device)["model_state"])

    os.makedirs(out_dir, exist_ok=True)

    dataset = CustomDataset(train=False, cache=True)
    model.eval()

    with torch.no_grad():
        for i in range(num_images):
            x, y_gt = dataset[i]          

            x_in = x.unsqueeze(0)   
            y_pred = model(x_in).squeeze(0)

            save_image(x, f"{out_dir}/input_{i}.png")
            save_image(y_gt, f"{out_dir}/gt_{i}.png")
            save_image(y_pred, f"{out_dir}/pred_{i}.png")

    print(f"Saved {num_images} comparisons to '{out_dir}/'")

#save_comparison_images(device)


# In[15]:


#test_inference_time(model, device=torch.device("cuda"), batch_size=1024)


# In[16]:


def compare_times(batch_sizes=None, devices=None):
    for dev in devices:
        model = Model()
        model.load_state_dict(torch.load("./weights.pt", map_location=device)["model_state"])

        for batch in batch_sizes:
            test_inference_time(model, dev, batch)


# In[18]:


#!jupyter nbconvert --to script carn_optional_v1.ipynb


# In[ ]:




