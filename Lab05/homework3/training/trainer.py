import torch
import torch.nn.functional as F
import time
from utils.logger import Logger
from utils.early_stopping import EarlyStopping
from torch import GradScaler
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import hflip
import torch.nn.functional as F

class BatchSizeScheduler:
    def __init__(self, dataloader, start_bs, max_bs, step_size, scale_factor=2):
        self.start_bs = start_bs
        self.max_bs = max_bs
        self.step_size = step_size
        self.scale_factor = scale_factor
        
        self.dataloader = dataloader
        self.current_bs = start_bs

    def step(self, epoch):
        if epoch != 0 and epoch % self.step_size == 0 and self.current_bs < self.max_bs:
            new_bs = min(int(self.current_bs * self.scale_factor), self.max_bs)
            self.dataloader = torch.utils.data.DataLoader(self.dataloader.dataset, batch_size=new_bs, 
                                                      shuffle=True, num_workers=self.dataloader.num_workers, pin_memory=True)
            print(f"Batch size updated: {self.current_bs} -> {new_bs}")
            self.current_bs = new_bs
            return True
        else:
            return False

def tta_translations(model, images, padding_size=2, offsets=[-2, 0, 2], flip=True):
    model.eval()
    image_size = 32
    padded = v2.functional.pad(images, [padding_size], fill=0.5)
    all_logits = []

    for i in offsets:
        for j in offsets:
            if i == 0 and j == 0:
                continue
            x = padding_size + i
            y = padding_size + j
            logits = model(padded[:, :, x:x + image_size, y:y + image_size])
            all_logits.append(logits)

            if flip:
                crop_flip = hflip(padded[:, :, x:x + image_size, y:y + image_size])
                logits_flip = model(crop_flip)
                all_logits.append(logits_flip)

    logits_center = model(images)
    all_logits.append(logits_center)

    if flip:
        images_flip = hflip(images)
        logits_flip = model(images_flip)
        all_logits.append(logits_flip)

    #final_logits = torch.mean(torch.stack(all_logits), dim=0)
    final_logits = torch.mean(torch.stack([logits.softmax(dim=1) for logits in all_logits]),dim=0)
    return final_logits


warmup_epochs = 5
def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, cfg, is_sam=False):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"
        
        self.model = model.to(self.device, non_blocking=True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.is_sam = is_sam

        self.enable_half = (self.device_type == "cuda")
        self.scaler = GradScaler(self.device_type, enabled=self.enable_half)
        
        self.logger = Logger(cfg)
        self.early_stop = EarlyStopping(patience=cfg['early_stopping'].get("patience", 10),
                                        min_delta=cfg['early_stopping'].get("delta", 0.0),
                                        mode=cfg['early_stopping'].get("mode", "min"))

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        n = 0

        for x,y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(self.device_type, enabled=self.enable_half):
                out = self.model(x)
                loss = F.cross_entropy(out, y,label_smoothing=0.2)
                #loss = F.cross_entropy(out, y)

            if self.is_sam:
                loss.backward()
                #self.scaler.scale(loss).backward()
                self.optimizer.first_step(zero_grad=True)

                with torch.autocast("cuda" if self.device == "cuda" else "cpu", enabled=True):
                    out2 = self.model(x)
                    loss2 = F.cross_entropy(out2, y,label_smoothing=0.2)
                    #loss2 = F.cross_entropy(out2, y)

                loss2.backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                #loss.backward()
                #self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)

        return total_loss / n, correct / n

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        n = 0

        with torch.no_grad():
            for x,y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.autocast(self.device_type, enabled=self.enable_half):
                    if self.cfg['dataset']['tta'] == True:
                        out = tta_translations(self.model, x)
                    else:
                        out = self.model(x)
                    loss = F.cross_entropy(out, y,label_smoothing=0.2)
                    #loss = F.cross_entropy(out, y)
                total_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                n += x.size(0)
        return total_loss / n, correct / n

    def fit(self, train_loader, val_loader, test_loader, epochs):
        if "batch_scheduler" in self.cfg and self.cfg['dataset']['cutmix_mixup']==False:
            bs_cfg = self.cfg["batch_scheduler"]
            batch_scheduler = BatchSizeScheduler(
                dataloader=train_loader,
                start_bs=bs_cfg.get("start_bs", self.cfg.get("batch_size", 32)),
                max_bs=bs_cfg.get("max_bs", 128),
                step_size=bs_cfg.get("step_size", 50),
                scale_factor=bs_cfg.get("scale_factor", 2),
            )
        else:
            batch_scheduler = None

        best_val = float('inf')

        for epoch in range(1, epochs+1):
            t0 = time.time()

            if self.cfg['optimizer']['warmup'] and epoch <= warmup_epochs:
                warmup_lr = float(self.cfg['optimizer']['lr']) * epoch / warmup_epochs
                set_lr(self.optimizer, warmup_lr)

            if batch_scheduler and (batch_scheduler.step(epoch - 1)):
                train_loader = batch_scheduler.dataloader

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            test_loss, test_acc = self.validate(test_loader)

            if self.scheduler:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if self.cfg.get("mode", "max") == "min": 
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - t0

            self.logger.log({"loss": train_loss, "acc": train_acc}, epoch, "train")
            self.logger.log({"loss": val_loss, "acc": val_acc}, epoch, "val")
            self.logger.log({"loss": test_loss, "acc": test_acc}, epoch, "test")
            self.logger.log({"lr": self.optimizer.param_groups[0]['lr']}, epoch, "lr")

            print(f"Epoch {epoch}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
                  f"time={epoch_time:.1f}s")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")

            if self.early_stop(val_loss if self.cfg['early_stopping'].get("mode", "min") == "min" else val_acc):
                print("Early stopping triggered")
                break

        self.logger.close()
