"""
ECLIPSE
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
The implementation is based on facebookresearch/Mask2Former.
"""

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import setup_mask_criterion


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            per_pixel: bool = False,
            softmask: bool = False,
            softcls: bool = True,
            # inference
            semantic_on: bool,
            panoptic_on: bool,
            instance_on: bool,
            mask_bg: bool,
            test_topk_per_image: int,
            # continual
            continual: bool = False,
            # prompt tuning
            num_prompts: int = 0, 
            backbone_freeze: bool = False,
            cls_head_freeze: bool = False,
            mask_head_freeze: bool = False,
            pixel_decoder_freeze: bool = False,
            query_embed_freeze: bool = False,
            trans_decoder_freeze: bool = False,
            prompt_mask_mlp: bool = False,
            prompt_no_obj_mlp: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            per_pixel: bool, whether to use matching loss or Cross-Entropy
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.mask_threshold = 0.5
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.per_pixel = per_pixel
        self.softmask = softmask
        self.softcls = softcls

        # continual
        self.continual = continual
        self.model_old = False

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.mask_bg = mask_bg
        self.test_topk_per_image = test_topk_per_image
        self.num_classes = sem_seg_head.num_classes
        
        # Parameters for ECLIPSE
        self.num_prompts = num_prompts
        self.backbone_freeze = backbone_freeze
        self.cls_head_freeze = cls_head_freeze
        self.mask_head_freeze = mask_head_freeze
        self.pixel_decoder_freeze = pixel_decoder_freeze
        self.query_embed_freeze = query_embed_freeze
        self.trans_decoder_freeze = trans_decoder_freeze
        self.prompt_mask_mlp = prompt_mask_mlp
        self.prompt_no_obj_mlp = prompt_no_obj_mlp

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
            
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        continual = hasattr(cfg, "CONT")
        if not continual:
            if not cfg.MODEL.MASK_FORMER.PER_PIXEL or not cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                # Loss parameters:
                criterion = setup_mask_criterion(cfg, sem_seg_head.num_classes)
            else:
                criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=meta.ignore_label)
        else:
            criterion = None

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                    or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                    or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "per_pixel": cfg.MODEL.MASK_FORMER.PER_PIXEL and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "softmask": cfg.MODEL.MASK_FORMER.SOFTMASK,
            "softcls": cfg.CONT.SOFTCLS,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "mask_bg": cfg.MODEL.MASK_FORMER.TEST.MASK_BG and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # continual
            "continual": continual,
            # prompt tuning
            "num_prompts": cfg.CONT.NUM_PROMPTS,
            "backbone_freeze": cfg.CONT.BACKBONE_FREEZE,
            "cls_head_freeze": cfg.CONT.CLS_HEAD_FREEZE,
            "mask_head_freeze": cfg.CONT.MASK_HEAD_FREEZE,
            "pixel_decoder_freeze": cfg.CONT.PIXEL_DECODER_FREEZE,
            "query_embed_freeze": cfg.CONT.QUERY_EMBED_FREEZE,
            "trans_decoder_freeze": cfg.CONT.TRANS_DECODER_FREEZE,
            "prompt_mask_mlp": cfg.CONT.PROMPT_MASK_MLP,
            "prompt_no_obj_mlp": cfg.CONT.PROMPT_NO_OBJ_MLP,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward_train_mask(self, images, batched_inputs, outputs):
        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = MaskFormer.prepare_targets(gt_instances, images.tensor.shape[-2:], per_pixel=self.per_pixel)
            # fixme: Assume, for semantic and panoptic, that dataset has background class
            if not self.mask_bg and (self.semantic_on or self.panoptic_on):
                for tar in targets:
                    tar['labels'] -= 1
        else:
            targets = None

        # bipartite matching-based loss
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses

    def forward_train_pixel(self, images, batched_inputs, outputs):
        losses = {}
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = MaskFormer.prepare_targets(targets, images.tensor.shape[-2:], per_pixel=self.per_pixel)
            targets = torch.cat(targets, dim=0)
            outputs_x = MaskFormer.prepare_semantic_train(outputs, targets, mask_bg=self.mask_bg)
            # downsample targets to reduce GPU memory consumption - performance are close.
            # This may introduce issues when having more than 255 classes. In that case, convert to float.
            targets = F.interpolate(targets.unsqueeze(0).byte(), size=outputs_x.shape[-2:], mode="nearest")[0].long()
            losses["loss_ce"] = self.criterion(outputs_x, targets)
            if "aux_outputs" in outputs:
                for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                    outputs_x = MaskFormer.prepare_semantic_train(aux_outputs, targets, mask_bg=self.mask_bg)
                    losses["loss_ce" + f"_{i}"] = self.criterion(outputs_x, targets)
        return losses

    def forward_inference(self, images, batched_inputs, outputs):
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"] if self.softmask else outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            if 'sem_seg' in input_per_image and self.continual and self.semantic_on:
                height, width = input_per_image['sem_seg'].shape
            else:
                height = input_per_image.get("height", image_size[0])  # image_size[0]
                width = input_per_image.get("width", image_size[1])  # image_size[1]
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                # That's interpolation to image size
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.model_old:
            return {"features": features, "outputs": outputs, "shape": images.tensor.shape[-2:]}

        if self.training:
            if self.criterion is None:
                return {"features": features, "outputs": outputs, "shape": images.tensor.shape[-2:]}
            else:
                if self.per_pixel:
                    return self.forward_train_pixel(images, batched_inputs, outputs)
                else:
                    return self.forward_train_mask(images, batched_inputs, outputs)
        else:
            return self.forward_inference(images, batched_inputs, outputs)

    @staticmethod
    def prepare_targets(targets, shape, per_pixel=False):
        h_pad, w_pad = shape
        new_targets = []
        if not per_pixel:
            for targets_per_image in targets:
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                           device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                    }
                )
        else:
            for gt_masks in targets:
                padded_masks = torch.zeros((1, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[0], : gt_masks.shape[1]] = gt_masks
                new_targets.append(padded_masks)
        return new_targets

    @staticmethod
    def prepare_semantic_train(outputs, targets, mask_bg=True):
        logits, mask = outputs["pred_logits"], outputs["pred_masks"]
        mask = mask.sigmoid()
        if mask_bg:
            semseg = torch.einsum("bqc,bqhw->bchw", logits, mask)
            semseg = semseg[:, :-1]  # Exclude no class since we have Bkg class
        else:
            raise NotImplementedError
        # out_size = targets.shape[-2:]
        # semseg = F.interpolate(semseg, size=out_size, mode="bilinear")
        return semseg

    def semantic_inference(self, mask_cls, mask_pred):
        mask_pred = mask_pred.sigmoid() if not self.softmask else mask_pred.softmax(dim=0)
        if self.per_pixel:
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)[:-1]
            semseg = torch.softmax(semseg, dim=0)
        elif self.mask_bg:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        else:
            # Assuming mask_pred is NHW (no batch size)
            if self.softcls:
                scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            else:
                scores, labels = mask_cls.sigmoid().max(-1)
                
            # mask_pred = mask_pred > 0.5

            h, w = mask_pred.shape[-2:]
            keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]  # sigmoid done up.

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            semseg = torch.zeros((h, w), dtype=torch.long, device=cur_masks.device)

            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                semseg = F.one_hot(semseg, self.num_classes+1).float().permute(2, 0, 1)
            else:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)

                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= self.mask_threshold).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= self.mask_threshold)
                    #fixme mask_area should be computed differently !
                    # mask_area = mask.sum().item()

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        semseg[mask] = pred_class + 1 # ADD ONE FOR BKG
                semseg = F.one_hot(semseg, self.num_classes+1).float().permute(2, 0, 1)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        if self.softcls:
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        else:
            scores, labels = mask_cls.sigmoid().max(-1)
            
        mask_pred = mask_pred.sigmoid() if not self.softmask else mask_pred.softmax(dim=0)

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item() + 1  # ADD ONE TO MATCH GT
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1] if self.softcls else mask_cls.sigmoid()[:, :-1]
        
        if self.softmask:
            mask_pred = mask_pred.softmax(dim=0)
            thr = 0.5
        else:
            thr = 0
            
        labels = torch.arange(
            self.sem_seg_head.num_classes, 
            device=self.device,
        ).unsqueeze(0).repeat(mask_pred.shape[0], 1).flatten(0, 1)
                
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > thr).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        if not self.softmask:
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                        result.pred_masks.flatten(1).sum(1) + 1e-6)
        else:
            mask_scores_per_image = (mask_pred.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def instance_inference2(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        n_queries = mask_pred.shape[0]
        
        # mask_cls : [N, C+1]
        # mask_pred : [N, H, W]
        
        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1] if self.softcls else mask_cls.sigmoid()[:, :-1]
        
        #---------------------------------------------------
        # mask_cls : [N, C+1]
        labels = mask_cls.sigmoid().argmax(-1) # [N, ]
        no_obj_scores = mask_cls.sigmoid()[:, -1] # [N, ]

        #keep = labels.ne(self.sem_seg_head.num_classes) & (no_obj_scores < 0.9)
        keep = torch.logical_not( (labels == self.sem_seg_head.num_classes) & (no_obj_scores > 0.9) )
        
        #keep = (no_obj_scores < 0.9)
        scores = scores[keep]
        #---------------------------------------------------
        
        # scores : [N, C]
        # max_s : [N, ]
#         max_s = scores.max(dim=-1)[0]
        
#         keep = (max_s > 0.05)
#         scores = scores[keep]

        if self.softmask:
            mask_pred = mask_pred.softmax(dim=0)
            thr = 0.5
        else:
            thr = 0
        mask_pred = mask_pred[keep]

        labels = torch.arange(
            self.sem_seg_head.num_classes, device=self.device
        ).unsqueeze(0).repeat(len(keep), 1).flatten(0, 1)
        # labels: [1, C] -> [N, C] -> [N*C]
        
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            min(scores.flatten(0, 1).shape[0], n_queries), sorted=False)
        labels_per_image = labels[topk_indices]
        # labels_per_image: [self.test_topk_per_image=100]
        
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # mask_pred: [N, H, W] -> [self.test_topk_per_image, H, W]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > thr).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        if not self.softmask:
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                        result.pred_masks.flatten(1).sum(1) + 1e-6)
        else:
            mask_scores_per_image = (mask_pred.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

        
    def freeze_for_prompt_tuning(self, ):
        """
        ECLIPSE: freeze parameters for visual prompt tuning
        """
        if not self.model_old:
            
            for k, p in self.named_parameters():
                p.requires_grad = False
                
            # unfreeze last classifier layer
            for k, p in self.sem_seg_head.predictor.class_embed.cls[-1].named_parameters():
                p.requires_grad = True
                
            # unfreeze cls-layer for no-obj
            for k, p in self.sem_seg_head.predictor.class_embed.cls[0].named_parameters():
                p.requires_grad = True
                
            if self.num_prompts > 0:
                # unfreeze prompt embeddings
                for k, p in self.sem_seg_head.predictor.prompt_feat[-1].named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.prompt_embed[-1].named_parameters():
                    p.requires_grad = True
                
            if not self.backbone_freeze:
                for k, p in self.backbone.named_parameters():
                    p.requires_grad = True
                    
            if not self.trans_decoder_freeze and self.num_prompts > 0:
                for k, p in self.sem_seg_head.predictor.prompt_transformer_self_attention_layers[-1].named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.prompt_transformer_cross_attention_layers[-1].named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.prompt_transformer_ffn_layers[-1].named_parameters():
                    p.requires_grad = True
                    
            if not self.pixel_decoder_freeze:
                for k, p in self.sem_seg_head.pixel_decoder.named_parameters():
                    p.requires_grad = True

            if not self.cls_head_freeze:
                for k, p in self.sem_seg_head.predictor.class_embed.cls.named_parameters():
                    p.requires_grad = True

            if not self.mask_head_freeze:
                for k, p in self.sem_seg_head.predictor.mask_embed.named_parameters():
                    p.requires_grad = True
                    
            if not self.query_embed_freeze:
                for k, p in self.sem_seg_head.predictor.query_feat.named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.query_embed.named_parameters():
                    p.requires_grad = True
                    
            if self.prompt_mask_mlp and self.num_prompts > 0:
                for k, p in self.sem_seg_head.predictor.prompt_mask_embed[-1].named_parameters():
                    p.requires_grad = True
                    
                for k, p in self.sem_seg_head.predictor.mask_embed.named_parameters():
                    p.requires_grad = False
                    
            if self.prompt_no_obj_mlp and self.num_prompts > 0:
                for k, p in self.sem_seg_head.predictor.prompt_no_obj_embed[-1].named_parameters():
                    p.requires_grad = True
                    
                for k, p in self.sem_seg_head.predictor.class_embed.cls[0].named_parameters():
                    p.requires_grad = False
                    
            if self.num_prompts == 0:
                # unfreeze prompt embeddings
                for k, p in self.sem_seg_head.predictor.query_feat.named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.query_embed.named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.mask_embed.named_parameters():
                    p.requires_grad = True
                
                
    def copy_prompt_embed_weights(self, ):
        if self.num_prompts > 0:
            base_feat_weights = self.sem_seg_head.predictor.query_feat.weight
            base_embed_weights = self.sem_seg_head.predictor.query_embed.weight
            
            base_feat_weights = base_feat_weights.mean(0, keepdims=True).repeat(self.num_prompts, 1)
            base_embed_weights = base_embed_weights.mean(0, keepdims=True).repeat(self.num_prompts, 1)
            
            self.sem_seg_head.predictor.prompt_feat[-1].load_state_dict({"weight": base_feat_weights})
            
            if isinstance(self.sem_seg_head.predictor.prompt_embed[-1], nn.ModuleList):
                embed_dict = {}
                for n in range(len(self.sem_seg_head.predictor.prompt_embed[-1])):
                    embed_dict[f"{n}.weight"] = base_embed_weights
            else:
                embed_dict = {"weight": base_embed_weights}
                
            self.sem_seg_head.predictor.prompt_embed[-1].load_state_dict(embed_dict)
                
            
    def copy_mask_embed_weights(self, ):
        if self.num_prompts > 0 and self.prompt_mask_mlp:
            self.sem_seg_head.predictor.prompt_mask_embed[-1].load_state_dict(
                self.sem_seg_head.predictor.mask_embed.state_dict()
            )
            
            
    def copy_prompt_trans_decoder_weights(self, ):
        if self.num_prompts > 0 and not self.trans_decoder_freeze:
            self.sem_seg_head.predictor.prompt_transformer_self_attention_layers[-1].load_state_dict(
                self.sem_seg_head.predictor.transformer_self_attention_layers.state_dict()
            )
            self.sem_seg_head.predictor.prompt_transformer_cross_attention_layers[-1].load_state_dict(
                self.sem_seg_head.predictor.transformer_cross_attention_layers.state_dict()
            )
            self.sem_seg_head.predictor.prompt_transformer_ffn_layers[-1].load_state_dict(
                self.sem_seg_head.predictor.transformer_ffn_layers.state_dict()
            )
            
            
    def copy_no_obj_weights(self, ):
        if self.num_prompts > 0:
            no_obj_weights = self.sem_seg_head.predictor.class_embed.cls[0].state_dict()
            novel_cls_weights = self.sem_seg_head.predictor.class_embed.cls[-1].state_dict()

            for k, v in no_obj_weights.items():
                if v.shape == novel_cls_weights[k].shape:
                    novel_cls_weights[k] = v
                else:
                    if "weight" in k:
                        novel_cls_weights[k] = v.repeat(novel_cls_weights[k].shape[0], 1)
                    elif "bias" in k:
                        novel_cls_weights[k] = v.repeat(novel_cls_weights[k].shape[0])

            self.sem_seg_head.predictor.class_embed.cls[-1].load_state_dict(novel_cls_weights)
