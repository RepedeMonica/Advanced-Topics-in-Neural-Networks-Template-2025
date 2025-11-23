import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(name, optimizer, cfg):
    scfg = cfg.get('scheduler', {})
    
    if name == "StepLR":
        step_size = int(scfg.get('step_size',10))  # Period of learning rate decay.
        gamma = float(scfg.get('gamma',0.1))       # Multiplicative factor of learning rate decay
        last_epoch = int(scfg.get('last_epoch',-1))
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
        
    if name == "ReduceLROnPlateau":
        factor=float(scfg.get('factor',0.1)) #  Factor by which the learning rate will be reduced
        mode=scfg.get("mode", "max")         # One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing
        patience=int(scfg.get('patience',10)) 
        threshold=float(scfg.get('threshold',1e-3)) 
        cooldown=int(scfg.get('cooldown',2)) 
        min_lr=float(scfg.get('min_lr',1e-7)) 
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=1e-3, cooldown=2, min_lr=1e-7)
    return None
