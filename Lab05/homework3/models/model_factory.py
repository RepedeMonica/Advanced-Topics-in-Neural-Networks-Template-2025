import timm
from .mlp import MLP

def create_model(cfg, num_classes):
    name = cfg['model']['name']
    pretrained = cfg['model'].get('pretrained', False)

    if name in ['resnet18','resnet50','resnest14d','resnest26d']:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        return model

    if name == 'mlp':
        mcfg = cfg['model']['mlp']
        return MLP(int(mcfg['input_dim']), mcfg['hidden'], num_classes)

    raise ValueError("Unknown model")
