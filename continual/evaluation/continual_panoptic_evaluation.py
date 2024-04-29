# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import cv2
import tempfile
import torch
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class ContinualCOCOPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, cfg, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        # get some info from config
        self._num_classes = 1 + cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
        self.base_classes = 1 + cfg.CONT.BASE_CLS
        self.novel_classes = cfg.CONT.INC_CLS * cfg.CONT.TASK

        self.old_classes = self.base_classes + (cfg.CONT.TASK-1) * cfg.CONT.INC_CLS \
            if cfg.CONT.TASK > 0 else 1 + cfg.CONT.BASE_CLS
        self.new_classes = cfg.CONT.INC_CLS if cfg.CONT.TASK > 0 else self.base_classes
        
        self._order = cfg.CONT.ORDER if cfg.CONT.ORDER is not None else list(range(1, self._num_classes))
        
        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)
            

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        #isthing = segment_info.pop("isthing", None)
        isthing = segment_info["isthing"]
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    
    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb, rgb2id

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    
                    # category_id re-ordering
                    pred_class = self._order[pred_class-1]
                    
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1
                
            else:
                # category_id re-ordering
                for p in segments_info:
                    p['category_id'] = self._order[p['category_id']-1]
                    p['isthing'] = (
                        p['category_id'] in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
                
            json_data["annotations"] = self._predictions
            
            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
#         for label, pqr in pq_res['per_class'].items():
#             res[f"PQ_c{label}"] = 100 * pqr['pq']
#             res[f"RQ_c{label}"] = 100 * pqr['rq']
#             res[f"SQ_c{label}"] = 100 * pqr['sq']

        pq = [p['pq'] for p in pq_res['per_class'].values()]
        rq = [p['rq'] for p in pq_res['per_class'].values()]
        sq = [p['sq'] for p in pq_res['per_class'].values()]
        
        tp = [p['tp'] if 'tp' in p else 0 for p in pq_res['per_class'].values()]
        fp = [p['fp'] if 'fp' in p else 0 for p in pq_res['per_class'].values()]
        fn = [p['fn'] if 'fn' in p else 0 for p in pq_res['per_class'].values()]

        res["PQ_base"] = 100 * np.sum(pq[1:self.base_classes]) / (self.base_classes-1)
        res["PQ_old"] = 100 * np.sum(pq[1:self.old_classes]) / (self.old_classes-1)
        res["PQ_new"] = 100 * np.sum(pq[self.old_classes:]) / self.new_classes
        res["PQ_novel"] = 100 * np.sum(pq[self.base_classes:]) / self.novel_classes if self.novel_classes > 0 else 0.

        res["RQ_base"] = 100 * np.sum(rq[1:self.base_classes]) / (self.base_classes-1)
        res["RQ_old"] = 100 * np.sum(rq[1:self.old_classes]) / (self.old_classes-1)
        res["RQ_new"] = 100 * np.sum(rq[self.old_classes:]) / self.new_classes
        res["RQ_novel"] = 100 * np.sum(rq[self.base_classes:]) / self.novel_classes if self.novel_classes > 0 else 0.

        res["SQ_base"] = 100 * np.sum(sq[1:self.base_classes]) / (self.base_classes-1)
        res["SQ_old"] = 100 * np.sum(sq[1:self.old_classes]) / (self.old_classes-1)
        res["SQ_new"] = 100 * np.sum(sq[self.old_classes:]) / self.new_classes
        res["SQ_novel"] = 100 * np.sum(sq[self.base_classes:]) / self.novel_classes if self.novel_classes > 0 else 0.
        
        res["TP_base"] = np.sum(tp[1:self.base_classes])
        res["TP_novel"] = np.sum(tp[self.base_classes:]) if self.novel_classes > 0 else 0.
        
        res["FP_base"] = np.sum(fp[1:self.base_classes])
        res["FP_novel"] = np.sum(fp[self.base_classes:]) if self.novel_classes > 0 else 0.
        
        res["FN_base"] = np.sum(fn[1:self.base_classes])
        res["FN_novel"] = np.sum(fn[self.base_classes:]) if self.novel_classes > 0 else 0.

        if "tp" in pq_res["All"]:
            res["TP"] = pq_res["All"]["tp"]
        if "fp" in pq_res["All"]:
            res["FP"] = pq_res["All"]["fp"]
        if "fn" in pq_res["All"]:
            res["FN"] = pq_res["All"]["fn"]
            
        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results
    

def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
        