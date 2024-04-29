# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator


class ContinualSemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        cfg,
        dataset_name,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self.output_file_iou = f"{output_dir}/iou.csv"
        self.output_file_acc = f"{output_dir}/acc.csv"

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None

        # get some info from config
        self._num_classes = 1 + cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
        self.base_classes = 1 + cfg.CONT.BASE_CLS
        self.novel_classes = cfg.CONT.INC_CLS * cfg.CONT.TASK

        self.old_classes = self.base_classes + (cfg.CONT.TASK-1) * cfg.CONT.INC_CLS \
            if cfg.CONT.TASK > 0 else 1 + cfg.CONT.BASE_CLS
        self.new_classes = cfg.CONT.INC_CLS if cfg.CONT.TASK > 0 else self.base_classes
        # Background is always present in evaluation, so add +1
        self._order = cfg.CONT.ORDER if cfg.CONT.ORDER is not None else list(range(1, self._num_classes))
        self._name = cfg.NAME
        self._task = cfg.CONT.TASK

        # assume class names has background and it's the first
        self._class_names = ['background']
        self._class_names += [meta.stuff_classes[x] for x in self._order[:self._num_classes-1]]  # sort class names
        self._ignore_label = meta.ignore_label

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):

            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int64)
            # with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
            #     gt = np.array(Image.open(f), dtype=np.int)
            gt = np.array(input['sem_seg'], dtype=np.int64)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float32)
        iou = np.full(self._num_classes, np.nan, dtype=np.float32)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float32)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float32)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float32)

        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        # prec[acc_valid] = tp[acc_valid] / (pos_pred[acc_valid] + 1e-5)
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        miou_base = np.sum(iou[1:self.base_classes]) / (self.base_classes-1)
        miou_old = np.sum(iou[1:self.old_classes]) / (self.old_classes-1)
        miou_new = np.sum(iou[self.old_classes:]) / self.new_classes
        miou_novel = np.sum(iou[self.base_classes:]) / self.novel_classes if self.novel_classes > 0 else 0.

        fg_iou = (np.sum(self._conf_matrix[1:-1, 1:-1]) + self._conf_matrix[0, 0]) / np.sum(self._conf_matrix[:-1, :-1])

        res = {}
        cls_iou = []
        cls_acc = []

        res["mIoU"] = 100 * miou
        res["mIoU_new"] = 100 * miou_new
        res["mIoU_novel"] = 100 * miou_novel
        res["mIoU_old"] = 100 * miou_old
        res["mIoU_base"] = 100 * miou_base

        res["fwIoU"] = 100 * fiou
        res["fgIoU"] = 100 * fg_iou
        for i, name in enumerate(self._class_names):
            #res["IoU-{}".format(name)] = 100 * iou[i]
            cls_iou.append(100 * iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            #res["ACC-{}".format(name)] = 100 * acc[i]
            cls_acc.append(100 * acc[i])

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})

        self._logger.info(results)
        self.print_on_file(results, cls_iou, cls_acc)

        #results["confusion_matrix"] = self.confusion_matrix_to_fig()
        return results

    def confusion_matrix_to_fig(self, norm_gt=True):
        cm = self._conf_matrix[:-1, :-1]
        if not norm_gt:
            div = (cm.sum(axis=1) + 0.000001)[:, np.newaxis]
        else:
            div = (cm.sum(axis=0) + 0.000001)[np.newaxis, :]
        fig = 1 - (cm.astype('float') / div)
        fig = torch.from_numpy(fig)
        fig = fig.repeat(3, 1, 1)
        fig[1] = 0.25
        fig[2] = 1-fig[2]

        return fig

    def print_on_file(self, results, cls_iou, cls_acc):
        with open(self.output_file_iou, "a") as out:
            out.write(f"{self._name},{self._task},")
            out.write(".".join([str(i) for i in cls_iou]))
        with open(self.output_file_acc, "a") as out:
            out.write(f"{self._name},{self._task},")
            out.write(".".join([str(i) for i in cls_acc]))

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list