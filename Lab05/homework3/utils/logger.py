from torch.utils.tensorboard import SummaryWriter
import wandb

class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_tb = cfg.get("tensorboard", True)
        self.use_wb = cfg.get("wandb", True)

        if self.use_tb:
            self.tb = SummaryWriter(log_dir=cfg.get("tb_dir", "runs/exp"))

        if self.use_wb:
            wandb.init(project=cfg.get("wandb_project", "training"), config=cfg)

    def log(self, metrics: dict, step: int, prefix="train"):
        for k, v in metrics.items():
            if self.use_tb:
                self.tb.add_scalar(f"{prefix}/{k}", v, step)
            if self.use_wb:
                wandb.log({f"{prefix}/{k}": v, "step": step})

    def close(self):
        if self.use_tb:
            self.tb.close()
        if self.use_wb:
            wandb.finish()
