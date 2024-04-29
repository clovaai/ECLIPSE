from typing import Mapping
import torch


class BaseDistillation:
    def __init__(self, cfg, model, model_old):
        self.cfg = cfg
        self.model = model
        self.model_old = model_old
        self.per_pixel = cfg.MODEL.MASK_FORMER.PER_PIXEL and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON
        self.use_bg = (cfg.MODEL.MASK_FORMER.TEST.MASK_BG or
                       cfg.MODEL.MASK_FORMER.PER_PIXEL) and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON

        self.classes = [cfg.CONT.BASE_CLS] + cfg.CONT.TASK * [cfg.CONT.INC_CLS]
        self.old_classes = cfg.CONT.BASE_CLS + (cfg.CONT.TASK-1) * cfg.CONT.INC_CLS if cfg.CONT.TASK > 0 else -1
        self.new_classes = cfg.CONT.INC_CLS if cfg.CONT.TASK > 0 else cfg.CONT.BASE_CLS
        self.num_classes = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
        if self.use_bg:
            self.num_classes += 1
            self.old_classes += 1
            self.classes[0] += 1

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @property
    def training(self):
        return self.model.training

    @property
    def device(self):
        return self.model.device

    def __call__(self, data) -> Mapping[str, torch.Tensor]:
        """
        This function should return a dictionary of losses.
        We'll return the one made by the model + the distillation we want to implement
        """
        raise NotImplementedError
