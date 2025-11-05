from pathlib import Path
import torch
from torchvision.io import write_png


def _make_synth(n: int, size: int = 128):
    x = torch.zeros(n, 3, size, size)
    for i in range(n):
        x[i] = torch.rand(3, size, size) * 0.2
        for _ in range(3):
            h = torch.randint(size // 8, size // 3, (1,)).item()
            w = torch.randint(size // 8, size // 3, (1,)).item()
            y = torch.randint(0, size - h, (1,)).item()
            x0 = torch.randint(0, size - w, (1,)).item()
            color = torch.rand(3, 1, 1) * 0.8 + 0.2
            x[i, :, y : y + h, x0 : x0 + w] = color
    return x


def ensure_or_create_synth_dataset(data_dir: str, size: int = 128, n_train: int = 96, n_val: int = 24):
    root = Path(data_dir)
    train = root / "train"
    val = root / "val"
    train.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)

    def has_images(p: Path):
        return any(p.glob("*.png")) or any(p.glob("*.jpg")) or any(p.glob("*.jpeg"))

    if has_images(train) and has_images(val):
        return

    # write synthetic images to disk as PNG using torchvision.io.write_png
    xtr = _make_synth(n_train, size)
    xva = _make_synth(n_val, size)
    for i in range(n_train):
        img = (xtr[i].clamp(0, 1) * 255).to(torch.uint8)
        write_png(img, str(train / f"{i:05d}.png"))
    for i in range(n_val):
        img = (xva[i].clamp(0, 1) * 255).to(torch.uint8)
        write_png(img, str(val / f"{i:05d}.png"))

