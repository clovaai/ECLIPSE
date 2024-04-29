from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from mask2former.modeling.criterion import calculate_uncertainty, SetCriterion
from mask2former.modeling.mask_losses import dice_loss_jit, sigmoid_ce_loss_jit, \
    softmax_dice_loss_jit, softmax_ce_loss_jit
from .loss import *


class KDSetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, deep_sup=True, focal=False,
                 old_classes=0, use_kd=False, uce=False, ukd=False, l2=False, use_bkg=False, kd_deep=True,
                 alpha=1., kd_reweight=False, kd_weight=0., mask_kd=0.,
                 ce_only_new=False, kd_use_novel=False, focal_alpha=10., focal_gamma=2., 
                 softcls=True):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses,
                         num_points, oversample_ratio, importance_sample_ratio, focal, focal_gamma, focal_alpha)
        self.old_classes = old_classes
        self.uce = uce and old_classes != 0
        self.ce_only_new = ce_only_new
        self.ukd = ukd
        self.l2 = l2
        self.use_kd = use_kd
        self.kd_deep = kd_deep
        self.alpha = alpha
        self.kd_use_novel = kd_use_novel
        self.deep_sup = deep_sup
        self.kd_reweight = kd_reweight
        self.kd_weight = kd_weight
        self.mask_kd = mask_kd
        self.softcls = softcls

        assert not (use_bkg and (uce or ukd)), "Using background mask is not supported with UCE or UKD distillation."

    def loss_labels(self, outputs, targets, indices, num_masks, outputs_old=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(  # BxN everything is NO_CLASS
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[idx] = target_classes_o  # put real classes inside (substitute NO_CLASS with real)

        if self.uce and self.focal:
            loss_ce = focal_uce_loss(src_logits.transpose(1, 2), target_classes,
                                     old_cl=self.old_classes,  gamma=self.focal_gamma, alpha=self.focal_alpha)
        elif self.uce:
            loss_ce = unbiased_cross_entropy_loss(src_logits.transpose(1, 2), target_classes,
                                                  old_cl=self.old_classes, weights=self.empty_weight, reduction='mean')
        elif self.focal:
            if self.softcls:
                loss_ce = focal_loss(src_logits.transpose(1, 2), target_classes,
                                     gamma=self.focal_gamma, alpha=self.focal_alpha)
            else:
                loss_ce = sigmoid_focal_loss(src_logits.transpose(1, 2), target_classes, 
                                             gamma=self.focal_gamma, alpha=self.focal_alpha)
                
        else:
            if self.softcls:
                loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight,
                                          ignore_index=255, reduction='mean')
            else:
                loss_ce = sigmoid_cross_entropy(src_logits, target_classes)
                

        losses = {"loss_ce": loss_ce}

        if outputs_old is not None and self.kd_weight > 0.:
            tar_logits = outputs_old["pred_logits"].float()
            if self.ukd:
                loss_kd = unbiased_knowledge_distillation_loss(src_logits.transpose(1, 2), tar_logits.transpose(1, 2),
                                                               reweight=self.kd_reweight, temperature=self.alpha)
            elif self.l2:
                loss_kd = L2_distillation_loss(src_logits.transpose(1, 2), tar_logits.transpose(1, 2))
            else:
                loss_kd = knowledge_distillation_loss(src_logits.transpose(1, 2), tar_logits.transpose(1, 2),
                                                      use_new=self.kd_use_novel, reweight=self.kd_reweight,
                                                      temperature=self.alpha)
            losses["loss_kd"] = loss_kd

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, outputs_old=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        if self.mask_kd > 0. and outputs_old is not None:
            new_masks = outputs["pred_masks"]
            old_masks = outputs_old["pred_masks"].detach()
            labels = old_masks.sigmoid()
            losses['loss_mask_kd'] = F.binary_cross_entropy_with_logits(new_masks, labels)

        del src_masks
        del target_masks
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks, outputs_old=None):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, outputs_old)

    def forward(self, outputs, targets, outputs_old=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                      For mask2Former (or this code), we need labels and masks
             outputs_old:  dict of tensors by old model , see the output specification of the model for the format
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "pred" in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, outputs_old))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs and self.deep_sup:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if outputs_old is not None and self.kd_deep:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks,
                                               outputs_old["aux_outputs"][i])
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class SoftmaxKDSetCriterion(KDSetCriterion):
    def loss_masks(self,  outputs, targets, indices, num_masks, outputs_old=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"].softmax(dim=1)  # Added softmax
        src_log_masks = torch.log_softmax(outputs["pred_masks"], dim=1)
        src_masks = src_masks[src_idx]
        src_log_masks = src_log_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        src_log_masks = src_log_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(2*logits-1),  # normalize to zero
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        point_log_logits = point_sample(
            src_log_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": softmax_ce_loss_jit(point_log_logits, point_labels, num_masks),
            "loss_dice": softmax_dice_loss_jit(point_logits, point_labels, num_masks),
        }

        if self.mask_kd > 0. and outputs_old is not None:
            new_masks = outputs["pred_masks"]
            old_masks = outputs_old["pred_masks"].detach()
            labels = old_masks.softmax(dim=1)
            output = torch.log_softmax(new_masks, dim=1)
            corr = torch.log_softmax(old_masks, dim=1)

            losses['loss_mask_kd'] = (labels * (corr - output)).mean()

        del src_masks
        del target_masks
        return losses