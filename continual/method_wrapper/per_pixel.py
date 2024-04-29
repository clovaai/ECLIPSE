from .base import BaseDistillation
import torch
import torch.nn as nn
from torch.nn import functional as F
from mask2former.maskformer_model import MaskFormer
from .pix_losses import UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, KnowledgeDistillationLoss, normalization
from detectron2.data import MetadataCatalog

from continual.modeling.pod import func_pod_loss

def pod_loss(output, output_old):
    # fixme actually pod is computed BEFORE ReLU. Due to Detectron, it is hard to move after...
    input_feat = output['features']
    input_feat = [input_feat[key] for key in ['res2', 'res3', 'res4', 'res5']] + [output['outputs']['features']]

    old_feat = output_old['features']
    old_feat = [old_feat[key].detach() for key in ['res2', 'res3', 'res4', 'res5']] + [output_old['outputs']['features']]

    loss = {"loss_pod": func_pod_loss(input_feat, old_feat, scales=[1, 2, 4])}
    return loss


class PerPixelDistillation(BaseDistillation):
    def __init__(self, cfg, model, model_old):
        super().__init__(cfg, model, model_old)
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        if cfg.CONT.DIST.UCE:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes,
                                                  reduction='mean', ignore_index=meta.ignore_label)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=meta.ignore_label)

        self.kd_loss = None

        if cfg.CONT.TASK > 0:
            if cfg.CONT.DIST.KD_WEIGHT > 0.:
                if cfg.CONT.DIST.UKD:
                    self.kd_loss = UnbiasedKnowledgeDistillationLoss(reduction='mean', alpha=cfg.CONT.DIST.ALPHA)
                else:
                    self.kd_loss = KnowledgeDistillationLoss(reduction="mean", alpha=cfg.CONT.DIST.ALPHA, use_new=cfg.CONT.DIST.USE_NEW)
                self.kd_weight = cfg.CONT.DIST.KD_WEIGHT

        self.kd_deep = cfg.CONT.DIST.KD_DEEP
        self.deep_sup = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        self.pod_weight = cfg.CONT.DIST.POD_WEIGHT
        self.pseudo = cfg.CONT.DIST.PSEUDO

        self.use_model_old = cfg.CONT.TASK > 0 and (self.kd_loss is not None or self.pod_weight > 0 or self.pseudo)

    def update_losses(self, losses, outputs, targets, outputs_old=None, suffix=""):
        outputs_x = MaskFormer.prepare_semantic_train(outputs, targets, mask_bg=self.use_bg)
        losses["loss_ce" + suffix] = self.criterion(outputs_x, targets)
        # Compute distillation for main output:
        if outputs_old is not None and self.kd_loss is not None:
            dist_target = MaskFormer.prepare_semantic_train(outputs_old, targets,
                                                            mask_bg=self.use_bg)
            losses["loss_kd" + suffix] = self.kd_weight * self.kd_loss(outputs_x, dist_target)

    def __call__(self, data):
        model_out = self.model(data)
        outputs = model_out['outputs']
        shape = model_out['shape']

        if self.use_model_old:
            model_out_old = self.model_old(data)
            outputs_old = model_out_old['outputs']
        else:
            model_out_old = None
            outputs_old = None

        losses = {}
        # XEntropy Loss - Classification // May also be UCE
        assert "sem_seg" in data[0], "Error: no SemSeg label in annotation!"
        targets = [x["sem_seg"].to(self.device) for x in data]
        targets = MaskFormer.prepare_targets(targets, shape, self.per_pixel)
        targets = torch.cat(targets, dim=0)
        # downsample targets to reduce GPU memory consumption - performance are close.
        # This may introduce issues when having more than 255 classes. In that case, convert to float.
        targets = F.interpolate(targets.unsqueeze(0).byte(), size=outputs['pred_masks'].shape[-2:], mode="nearest")[0].long()

        if self.pseudo and self.use_model_old:
            mask_background = targets == 0
            pred_old = MaskFormer.prepare_semantic_train(model_out_old['outputs'], targets, mask_bg=self.use_bg)  # BC_oldHW
            probs_old = torch.softmax(pred_old, dim=1)
            pseudo_labels = probs_old.argmax(dim=1)
            pseudo_labels[probs_old.max(dim=1)[0] < 0.8] = 255
            targets[mask_background] = pseudo_labels[mask_background]

        self.update_losses(losses, outputs, targets, outputs_old)
        if self.pod_weight > 0 and self.use_model_old:
            losses.update(pod_loss(model_out, model_out_old))

        if "aux_outputs" in outputs and self.deep_sup:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if outputs_old is not None and self.kd_deep:
                    self.update_losses(losses, aux_outputs, targets, outputs_old["aux_outputs"][i], suffix=f"_{i}")
                else:
                    self.update_losses(losses, aux_outputs, targets, suffix=f"_{i}")

        return losses
