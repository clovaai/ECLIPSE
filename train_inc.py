"""
ECLIPSE
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
The implementation is based on fcdl94/CoMFormer and facebookresearch/Mask2Former.
"""

"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import warnings
warnings.filterwarnings('ignore')

import copy
import itertools
import logging
import wandb
import os
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Set
import torch
from fvcore.nn.precise_bn import get_bn_modules

# Detectron
from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts, DatasetMapper
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluators,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
    COCOEvaluator,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer, TrainerBase
from detectron2.engine import hooks
from detectron2.engine.defaults import create_ddp_model, default_writers

# MaskFormer
from mask2former import (
    InstanceSegEvaluator,
    #COCOInstanceNewBaselineDatasetMapper,
    #MaskFormerInstanceDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    MaskFormerPanopticDatasetMapper,
)

from continual import add_continual_config
from continual.data import ContinualDetectron, InstanceContinualDetectron
from continual.evaluation import ContinualSemSegEvaluator, ContinualCOCOPanopticEvaluator, ContinualCOCOEvaluator
from continual.method_wrapper import build_wrapper
from continual.utils.hooks import BetterPeriodicCheckpointer, BetterEvalHook
from continual.modeling.classifier import WA_Hook
from continual.data import MaskFormerInstanceDatasetMapper, COCOInstanceNewBaselineDatasetMapper

# print('done')


class IncrementalTrainer(TrainerBase):
    """
    Extension of the Trainer class adapted to Continual MaskFormer.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)

        self.model_old = self.build_model(cfg, old=True) if cfg.CONT.TASK > 0 and cfg.CONT.NUM_PROMPTS == 0 else None

        self.optimizer = optimizer = self.build_optimizer(cfg, model)
        self.data_loader = data_loader = self.build_train_loader(cfg)

        self.model = model = create_ddp_model(model, broadcast_buffers=False)
        model_wrapper = build_wrapper(cfg, model, self.model_old)

        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model_wrapper, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        if self.model_old is not None:
            self.checkpointer_old = DetectionCheckpointer(self.model_old, cfg.OUTPUT_DIR)
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self._last_eval_results = None
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        
        """
        ECLIPSE: initialization
        """
        if self.cfg.CONT.NUM_PROMPTS > 0:
            self.model.module.copy_prompt_embed_weights()
            self.model.module.copy_mask_embed_weights()
            self.model.module.copy_no_obj_weights()
            self.model.module.copy_prompt_trans_decoder_weights()
        
        if self.model_old is not None:
            # We never want to resume it! Resume = False even when loading from checkpoint.
            # This should be strict. If you see that the model does not use some parameters, there's an error.
            self.checkpointer_old.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=False)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_hooks(self):
        """
        Taken from DefaultTrainer (detectron2.engine.defaults)

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(BetterPeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(BetterEvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))

        if self.cfg.CONT.WA_STEP > 0 and self.cfg.CONT.TASK > 0:
            ret.append(WA_Hook(model=self.model, step=100, distributed=True))

        return ret

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
        self.write_results(self._last_eval_results)
        return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    @classmethod
    def build_model(cls, cfg, old=False):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        if old:
            cfg = cfg.clone()
            cfg.defrost()
            cfg.CONT.TASK -= 1
        model = build_model(cfg)
        if not old:
            logger = logging.getLogger(__name__)
            logger.info("Model:\n{}".format(model))
            
            if (cfg.CONT.TASK > 0 and cfg.CONT.NUM_PROMPTS > 0 and cfg.CONT.BACKBONE_FREEZE):
                logger.info("Model Freezing for Visual Prompt Tuning")
                model.freeze_for_prompt_tuning()
            
        else:
            model.model_old = True  # we need to set this as the old model.
            model.sem_seg_head.predictor.set_as_old_model()
            model.eval()  # and we set it to eval mode.
            for par in model.parameters():
                par.requires_grad = False
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        
        logger = logging.getLogger(__name__)
        unfreeze_params = []
        freeze_params = []
        
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    freeze_params.append(f"{module_name}.{module_param_name}")
                    continue
                    
                unfreeze_params.append(f"{module_name}.{module_param_name}")
                
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "sem_seg_head.predictor.mask_embed" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.HEAD_MULTIPLIER
                # if "sem_seg_head.predictor.class_embed" in module_name:
                #     hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.HEAD_MULTIPLIER
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    # print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                    # hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.HEAD_MULTIPLIER
                params.append({"params": [value], **hyperparams})

        if comm.is_main_process():
            print("\n============ freeze parameters ============")
            for param in freeze_params:
                print(f"   {param}")
            print("============ freeze parameters ============\n")

            print("\n============ unfreeze parameters ============")
            for param in unfreeze_params:
                print(f"   {param}")
            print("============ unfreeze parameters ============\n")
            
            num_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            storage_size = sum(
                p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
            ) / (1024 ** 2)  # Convert to MB
            
            print(f"============ num_params: {num_params//1000} K" )
            print(f"============ storage_size: {storage_size} MB" )
                
        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.projects.deeplab.build_lr_scheduler`.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":  # and "voc" in cfg.DATASETS.TRAIN[0]:
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            wrapper = ContinualDetectron
            n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - 1
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            wrapper = InstanceContinualDetectron
            n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            wrapper = InstanceContinualDetectron
            n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            wrapper = ContinualDetectron
            n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - 1  # we have bkg
        else:
            raise NotImplementedError("At the moment, we support only segmentation")

        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        scenario = wrapper(
            dataset,
            # Continuum related:
            initial_increment=cfg.CONT.BASE_CLS, increment=cfg.CONT.INC_CLS,
            nb_classes=n_classes,
            save_indexes=os.getenv("DETECTRON2_DATASETS", "datasets") + '/' + cfg.TASK_NAME,
            mode=cfg.CONT.MODE, class_order=cfg.CONT.ORDER,
            # Mask2Former related:
            mapper=mapper, cfg=cfg
        )
        return scenario[cfg.CONT.TASK]

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        if not hasattr(cls, "scenario"):
            mapper = DatasetMapper(cfg, False)
            if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
                wrapper = ContinualDetectron
                n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - 1
            elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
                mapper = COCOInstanceNewBaselineDatasetMapper(cfg, False)
                wrapper = InstanceContinualDetectron
                n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
                mapper = MaskFormerInstanceDatasetMapper(cfg, False)
                wrapper = InstanceContinualDetectron
                n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
                mapper = MaskFormerPanopticDatasetMapper(cfg, False)
                wrapper = ContinualDetectron
                n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - 1  # we have bkg
            else:
                raise NotImplementedError("At the moment, we support only segmentation")

            dataset = get_detection_dataset_dicts(
                dataset_name,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                proposal_files=None,
            )
            scenario = wrapper(
                dataset,
                # Continuum related:
                initial_increment=cfg.CONT.BASE_CLS, increment=cfg.CONT.INC_CLS,
                nb_classes=n_classes,
                save_indexes=os.getenv("DETECTRON2_DATASETS", "datasets") + '/' + cfg.TASK_NAME,
                mode=cfg.CONT.MODE, class_order=cfg.CONT.ORDER,
                # Mask2Former related:
                mapper=mapper, cfg=cfg, masking_value=0,
            )
            cls.scenario = scenario[cfg.CONT.TASK]
        else:
            print("Using computed scenario.")
        return cls.scenario

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg"]:  # , "ade20k_panoptic_seg"]:
            evaluator_list.append(
                ContinualSemSegEvaluator(
                    cfg,
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(ContinualCOCOEvaluator(cfg, dataset_name, output_dir=output_folder))
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
            evaluator_list.append(ContinualCOCOPanopticEvaluator(cfg, dataset_name, output_folder))
            
        if evaluator_type in [
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
            "coco_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(ContinualCOCOPanopticEvaluator(cfg, dataset_name, output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        Taken from DefaultTrainer (detectron2.engine.defaults)
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
                cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def state_dict(self):
        ret = super().state_dict()
        ret['trainer'] = self._trainer.state_dict()

        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict['trainer'])

    def write_results(self, results):
        name = self.cfg.NAME
        path = f"results/{self.cfg.TASK_NAME}.csv"
        path_acc = f"results/{self.cfg.TASK_NAME}_acc.csv"
        if "sem_seg" in results:
            res = results['sem_seg']
            cls_iou = []
            cls_acc = []
            for k in res:
                if k.startswith("IoU-"):
                    cls_iou.append(res[k])
                if k.startswith("ACC-"):
                    cls_acc.append(res[k])

            with open(path, "a") as out:
                out.write(f"{name},{self.cfg.CONT.TASK},{res['mIoU_base']},{res['mIoU_novel']},{res['mIoU']},")
                out.write(",".join([str(i) for i in cls_iou]))
                out.write("\n")

            with open(path_acc, "a") as out:
                out.write(f"{name},{self.cfg.CONT.TASK},{res['mACC']},{res['pACC']},-,")
                out.write(",".join([str(i) for i in cls_acc]))
                out.write("\n")
        if 'panoptic_seg' in results:
            res = results['panoptic_seg']
            cls_pq = OrderedDict()
            cls_rq = OrderedDict()
            cls_sq = OrderedDict()
            for k in res:
                if k.startswith("PQ_c"):
                    cls_pq[int(k[4:])] = res[k]
                if k.startswith("RQ_c"):
                    cls_rq[int(k[4:])] = res[k]
                if k.startswith("SQ_c"):
                    cls_sq[int(k[4:])] = res[k]
            with open(path, "a") as out:
                out.write(f"{name},{self.cfg.CONT.TASK},{res['PQ']},{res['RQ']},{res['SQ']},")
                out.write(",".join([str(i) for i in cls_pq.values()]))
                out.write(f",")
                out.write(",".join([str(i) for i in cls_rq.values()]))
                out.write(f",")
                out.write(",".join([str(i) for i in cls_sq.values()]))
                out.write("\n")
        if 'segm' in results:
            res = results['segm']
            path = f"results/{self.cfg.TASK_NAME}.csv"
            class_ap = []
            for k in res:
                if k.startswith("AP-"):
                    class_ap.append(res[k])
            with open(path, "a") as out: # "AP", "AP50", "AP75", "APs", "APm", "APl"
                out.write(f"{name},{self.cfg.CONT.TASK},{res['AP']},{res['AP50']},{res['AP75']},")
                out.write(",".join([str(i) for i in class_ap]))
                out.write("\n")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.NAME = "Exp"

    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_continual_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
        suffix = "-pan"
    elif cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        suffix = "-ins"
    else:
        suffix = ""

    if cfg.CONT.MODE == 'overlap':
        cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:3]}{suffix}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-ov"
    elif cfg.CONT.MODE == "disjoint":
        cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:3]}{suffix}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-dis"
    else:
        cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:3]}{suffix}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-seq"

    if cfg.CONT.ORDER_NAME is not None:
        cfg.TASK_NAME += "-" + cfg.CONT.ORDER_NAME

    cfg.OUTPUT_ROOT = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "/" + cfg.TASK_NAME + "/" + cfg.NAME + "/step" + str(cfg.CONT.TASK)

    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    if comm.get_rank() == 0 and cfg.WANDB:
        wandb.init(project="ECLIPSE", entity="your_entity",
                   name=f"{cfg.NAME}_step_{cfg.CONT.TASK}",
                   config=cfg, sync_tensorboard=True, group="PVT_"+cfg.TASK_NAME, settings=wandb.Settings(start_method="fork"))

    return cfg


def main(args):
    cfg = setup(args)

    if hasattr(cfg, 'CONT') and cfg.CONT.TASK > 0:
        cfg.defrost()
        if cfg.CONT.WEIGHTS is None:  # load from last step
            cfg.MODEL.WEIGHTS = cfg.OUTPUT_ROOT + "/" + cfg.TASK_NAME + "/" + cfg.NAME + f"/step{cfg.CONT.TASK - 1}/model_final.pth"
        else:  # load from cfg
            cfg.MODEL.WEIGHTS = cfg.CONT.WEIGHTS

        cfg.freeze()

    if args.eval_only:
        model = IncrementalTrainer.build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = IncrementalTrainer.test(cfg, model)
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(IncrementalTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = IncrementalTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    ret = trainer.train()
    if comm.get_rank() == 0:
        wandb.finish()
    return ret


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
