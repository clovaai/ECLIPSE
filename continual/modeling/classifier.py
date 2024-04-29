"""
ECLIPSE
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
The implementation is based on fcdl94/CoMFormer and facebookresearch/Mask2Former.
"""

from torch import nn
import torch
from torch.nn import functional as F
from detectron2.engine.hooks import HookBase

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

    
class WA_Hook(HookBase):
    def __init__(self, model, step, distributed=True):
        self.classifier = None
        if distributed:
            self.classifier = model.module.sem_seg_head.predictor.class_embed.cls
        else:
            self.classifier = model.sem_seg_head.predictor.class_embed.cls
        self.step = step
        self.iteration = 0

    def after_step(self):
        if self.trainer.iter % self.step == 0:
            with torch.no_grad():
                new_cls = self.classifier[-1].weight
                old_cls = torch.cat([c.weight for c in self.classifier[1:-1]], dim=0)
                norm_new = torch.norm(new_cls, dim=1)
                norm_old = torch.norm(old_cls, dim=1)
                gamma = torch.mean(norm_old) / torch.mean(norm_new)
                self.classifier[-1].weight.mul_(gamma)


class IncrementalClassifier(nn.Module):
    def __init__(self, classes, norm_feat=False, channels=256, bias=True, deep_cls=False):
        super().__init__()
        
        if deep_cls:
            self.cls = nn.ModuleList(
                [MLP(channels, channels, c, 3) for c in classes])
            
        else:
            self.cls = nn.ModuleList(
                [nn.Linear(channels, c, bias=bias) for c in classes])
        
        self.norm_feat = norm_feat

    def forward(self, x):
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=3)
        out = []
        for mod in self.cls[1:]:
            out.append(mod(x))
        out.append(self.cls[0](x))  # put as last the void class
        return torch.cat(out, dim=2)


class CosineClassifier(nn.Module):
    def __init__(self, classes, norm_feat=True, channels=256, scaler=10.):
        super().__init__()
        self.cls = nn.ModuleList(
            [nn.Linear(channels, c, bias=False) for c in classes])
        self.norm_feat = norm_feat
        self.scaler = scaler

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = []
        for mod in self.cls[1:]:
            out.append(self.scaler * F.linear(x, F.normalize(mod.weight, dim=1, p=2)))
        out.append(self.scaler * F.linear(x, F.normalize(self.cls[0].weight, dim=1, p=2)))  # put as last the void class
        return torch.cat(out, dim=2)
