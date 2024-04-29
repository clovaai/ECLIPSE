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


class PseudoSetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, deep_sup=True, use_kd=False,
                 old_classes=0, use_bkg=False, alpha=1., weight_masks=1., focal=False, smooth=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses,
                         num_points, oversample_ratio, importance_sample_ratio)
        self.old_classes = old_classes
        self.use_bkg = use_bkg
        self.alpha = alpha
        self.weight_masks = weight_masks
        self.kd = use_kd
        self.focal = focal
        self.deep_sup = deep_sup
        self.label_smoothing = smooth

    def loss_labels(self, outputs, targets, indices, num_masks, outputs_old=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)  # match ground truth with query ID

        # BxQ this vector associate each query ID with the corr. class
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, self.num_classes+1) * 0.1  # make it one_hot! -> B x Q x K+1

        if "probs" in targets[0] and self.kd:  # Use Soft Pseudo-labels
            target_classes_s = torch.cat([t["probs"][J] for t, (_, J) in zip(targets, indices)])
            target[idx] = target_classes_s
        else:  # Use Hard Pseudo-labels
            target[idx] *= 10.  # so that GT goes back to 1.

        weights = target.sum(dim=-1)  # B x Q: for no_class is 0.1, for other is 1 (softmax sum)
        # CE Loss is already weighted by the targets!
        outputs = torch.log_softmax(src_logits.transpose(1, 2), dim=1)
        loss_ce = - (outputs * target.transpose(1, 2)).sum(dim=1)  # B, Q
        if self.focal:
            alpha = 10
            gamma = 2
            pt = torch.exp(-loss_ce)
            loss_ce = alpha * (1 - pt) ** gamma * loss_ce
        else:
            loss_ce = (loss_ce.sum(dim=1) / weights.sum(dim=1)).mean()

        losses = {"loss_ce": loss_ce}

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


class SoftmaxPseudoSetCriterion(PseudoSetCriterion):
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
