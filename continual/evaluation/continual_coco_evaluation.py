# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json, _evaluate_predictions_on_coco

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval

class ContinualCOCOEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        cfg,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
            allow_cached_coco (bool): Whether to use cached coco json from previous validation
                runs. You should set this to False if you need to use different validation data.
                Defaults to True.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        if use_fast_impl and (COCOeval_opt is COCOeval):
            self._logger.info("Fast COCO eval is not built. Falling back to official COCO eval.")
            use_fast_impl = False
        self._use_fast_impl = use_fast_impl

        # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
        # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
        # 3rd element (100) is used as the limit on the number of detections per image when
        # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
        # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]
        self._max_dets_per_image = max_dets_per_image

        if tasks is not None and isinstance(tasks, CfgNode):
            kpt_oks_sigmas = (
                tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
            )
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path, allow_cached=allow_cached_coco)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        if self._do_evaluation:
            self._kpt_oks_sigmas = kpt_oks_sigmas
            
        #--------------------------------------------------------------------------
        # get some info from config
        self.class_order = cfg.CONT.ORDER

        self._num_classes = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
        self.base_classes = cfg.CONT.BASE_CLS
        self.novel_classes = cfg.CONT.INC_CLS * cfg.CONT.TASK

        self.old_classes = self.base_classes + (cfg.CONT.TASK-1) * cfg.CONT.INC_CLS \
            if cfg.CONT.TASK > 0 else cfg.CONT.BASE_CLS
        self.new_classes = cfg.CONT.INC_CLS if cfg.CONT.TASK > 0 else self.base_classes

        self.base_cls_idx = self.class_order[:self.base_classes]
        self.novel_cls_idx = self.class_order[self.base_classes:]
        self.old_cls_idx = self.class_order[:self.old_classes]
        self.new_cls_idx = self.class_order[self.old_classes:]


    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
                
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
                
            if len(prediction) > 1:
                self._predictions.append(prediction)
                
    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)
                
#         if "proposals" in predictions[0]:
#             self._eval_box_proposals(predictions)
            
        self._results = OrderedDict()
        
        if len(predictions) > 0 and "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
            
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
    

    def _eval_predictions(self, predictions, img_ids=None, prefix=""):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # beom : class ordering
        for result in coco_results:
            category_id = result["category_id"]
            result["category_id"] = self.class_order[category_id]

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            if task != "segm":
                continue
            
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    cocoeval_fn=COCOeval_opt if self._use_fast_impl else COCOeval,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            
            if task not in self._results:
                self._results[task] = {}
            
            self._results[task].update({k+prefix: v for k, v in res.items()})
            
    
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_per_category_iou50 = []
        results_per_category_iou75 = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
            
            precision_iou50 = precisions[0, :, idx, 0, -1]
            precision_iou50 = precision_iou50[precision_iou50 > -1]
            ap50 = np.mean(precision_iou50) if precision_iou50.size else float("nan")
            results_per_category_iou50.append(("{}".format(name), float(ap50 * 100)))
            
            precision_iou75 = precisions[5, :, idx, 0, -1]
            precision_iou75 = precision_iou75[precision_iou75 > -1]
            ap75 = np.mean(precision_iou75) if precision_iou75.size else float("nan")
            results_per_category_iou75.append(("{}".format(name), float(ap75 * 100)))
            
        ap_per_category = np.array([k[1] for k in results_per_category])
        ap50_per_category = np.array([k[1] for k in results_per_category_iou50])
        ap75_per_category = np.array([k[1] for k in results_per_category_iou75])

        AP_old = ap_per_category[self.old_cls_idx].mean()
        AP_new = ap_per_category[self.new_cls_idx].mean()
        AP_base = ap_per_category[self.base_cls_idx].mean()
        AP_novel = ap_per_category[self.novel_cls_idx].mean()

        AP50_old = ap50_per_category[self.old_cls_idx].mean()
        AP50_new = ap50_per_category[self.new_cls_idx].mean()
        AP50_base = ap50_per_category[self.base_cls_idx].mean()
        AP50_novel = ap50_per_category[self.novel_cls_idx].mean()

        AP75_old = ap75_per_category[self.old_cls_idx].mean()
        AP75_new = ap75_per_category[self.new_cls_idx].mean()
        AP75_base = ap75_per_category[self.base_cls_idx].mean()
        AP75_novel = ap75_per_category[self.novel_cls_idx].mean()

        # AP_old = sum([k[1] for k in results_per_category[:self.old_classes]]) / self.old_classes
        # AP_new = sum([k[1] for k in results_per_category[self.old_classes:]]) / self.new_classes
        # AP_base = sum([k[1] for k in results_per_category[:self.base_classes]]) / self.base_classes
        # AP_novel = sum([k[1] for k in results_per_category[self.base_classes:]]) / self.novel_classes if self.novel_classes > 0 else 0.
        
        # AP50_old = sum([k[1] for k in results_per_category_iou50[:self.old_classes]]) / self.old_classes
        # AP50_new = sum([k[1] for k in results_per_category_iou50[self.old_classes:]]) / self.new_classes
        # AP50_base = sum([k[1] for k in results_per_category_iou50[:self.base_classes]]) / self.base_classes
        # AP50_novel = sum([k[1] for k in results_per_category_iou50[self.base_classes:]]) / self.novel_classes if self.novel_classes > 0 else 0.
        
        # AP75_old = sum([k[1] for k in results_per_category_iou75[:self.old_classes]]) / self.old_classes
        # AP75_new = sum([k[1] for k in results_per_category_iou75[self.old_classes:]]) / self.new_classes
        # AP75_base = sum([k[1] for k in results_per_category_iou75[:self.base_classes]]) / self.base_classes
        # AP75_novel = sum([k[1] for k in results_per_category_iou75[self.base_classes:]]) / self.novel_classes if self.novel_classes > 0 else 0.

        results.update({"AP_old": AP_old, "AP_new": AP_new, "AP_base": AP_base, "AP_novel": AP_novel})
        results.update({"AP50_old": AP50_old, "AP50_new": AP50_new, "AP50_base": AP50_base, "AP50_novel": AP50_novel})
        results.update({"AP75_old": AP75_old, "AP75_new": AP75_new, "AP75_base": AP75_base, "AP75_novel": AP75_novel})
        
        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        #results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results