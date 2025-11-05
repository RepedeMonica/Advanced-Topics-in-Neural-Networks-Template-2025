import os
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderFlat(Dataset):
    """
    Simple flat image dataset:
    - expects PNG/JPG files under root directory (non-recursive by default)
    - returns image_tensor (no label)
    Use torchvision transforms to produce tensors.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        recursive: bool = False,
        extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ) -> None:
        self.root = root
        self.transform = transform
        self.extensions = extensions
        self.files = []
        if recursive:
            for dirpath, _, filenames in os.walk(root):
                for f in filenames:
                    if f.lower().endswith(self.extensions):
                        self.files.append(os.path.join(dirpath, f))
        else:
            for f in os.listdir(root):
                if f.lower().endswith(self.extensions):
                    self.files.append(os.path.join(root, f))
        self.files.sort()
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No images with extensions {self.extensions} found in {root}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im


def get_datasets(
    data_dir: str,
    image_size: int = 256,
    center_crop: bool = True,
    recursive: bool = False,
):
    """Build train/val datasets given a root with train/ and val/ subfolders."""
    from torchvision import transforms as T

    tx = []
    if center_crop:
        tx.append(T.CenterCrop(image_size))
    tx.extend([T.Resize((image_size, image_size)), T.ToTensor()])
    # Normalize to [0,1] only; models handle further scaling if needed.
    transform = T.Compose(tx)

    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")
    train_ds = ImageFolderFlat(train_root, transform=transform, recursive=recursive)
    val_ds = ImageFolderFlat(val_root, transform=transform, recursive=recursive)
    return train_ds, val_ds
