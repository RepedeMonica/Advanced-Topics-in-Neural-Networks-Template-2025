import torch.optim as optim
from muon import Muon, MuonWithAuxAdam
from .sams import SAM

def get_base_optimizer(name, params, cfg):
    ocfg = cfg['optimizer']
    lr = float(ocfg.get('lr', 0.005))
    wd = float(ocfg.get('weight_decay', 0.0005))

    if name == "SGD":
        return optim.SGD(params, lr=lr, momentum=ocfg.get('momentum', 0.99), nesterov=ocfg.get('nesterov', False), weight_decay=wd)

    if name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=float(ocfg.get('eps', 1e-8)))

    if name == "AdamW":
        return optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=float(ocfg.get('eps', 1e-8)))

    if name == "Muon":
        # return Muon(list(params), lr=lr, weight_decay=wd)
        hidden_weights = [p for p in params if p.ndim >= 2]
        other_params = [p for p in params if p.ndim < 2]

        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=lr, weight_decay=wd),
            dict(params=other_params, use_muon=False, lr=lr, betas=(0.9, 0.95), weight_decay=wd)
        ]

        optimizer = MuonWithAuxAdam(param_groups)
        return optimizer

    raise ValueError("Unknown optimizer")


def get_optimizer(name, model, cfg):
    ocfg = cfg["optimizer"]

    if name == "SAM":
        base_name = ocfg.get("base", "SGD")
        lr = float(ocfg['lr'])
        wd = float(ocfg.get('weight_decay', 5e-4))
        rho = float(ocfg.get('rho', 0.05))

        if base_name == "SGD":
            base_cls = optim.SGD
            extra = dict(momentum=ocfg.get("momentum", 0.9),
                         nesterov=ocfg.get("nesterov", True))
        else:
            raise ValueError(f"Unknown SAM base optimizer: {base_name}")
        return SAM(model.parameters(), base_cls, rho=rho, lr=lr, weight_decay=wd, **extra)

    else:
        return get_base_optimizer(name, model.parameters(), cfg)
