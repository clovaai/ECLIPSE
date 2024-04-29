from detectron2.config import CfgNode as CN


def add_continual_config(cfg):
    cfg.WANDB = True

    cfg.CONT = CN()
    cfg.CONT.BASE_CLS = 15
    cfg.CONT.INC_CLS = 5
    cfg.CONT.ORDER = list(range(1, 21))
    cfg.CONT.ORDER_NAME = None
    cfg.CONT.TASK = 0
    cfg.CONT.WEIGHTS = None
    cfg.CONT.MODE = "overlap"  # Choices "overlap", "disjoint", "sequential"
    cfg.CONT.INC_QUERY = False
    cfg.CONT.COSINE = False
    cfg.CONT.USE_BIAS = True
    cfg.CONT.WA_STEP = 0

    cfg.CONT.DIST = CN()
    cfg.CONT.DIST.POD_WEIGHT = 0.
    cfg.CONT.DIST.KD_WEIGHT = 0.
    cfg.CONT.DIST.ALPHA = 1.
    cfg.CONT.DIST.UCE = False
    cfg.CONT.DIST.UKD = False
    cfg.CONT.DIST.L2 = False
    cfg.CONT.DIST.KD_REW = False
    cfg.CONT.DIST.KD_DEEP = False
    cfg.CONT.DIST.USE_NEW = False
    cfg.CONT.DIST.EOS_POW = 0.
    cfg.CONT.DIST.CE_NEW = False
    cfg.CONT.DIST.PSEUDO = False
    cfg.CONT.DIST.PSEUDO_TYPE = 0
    cfg.CONT.DIST.IOU_THRESHOLD = 0.5
    cfg.CONT.DIST.PSEUDO_THRESHOLD = 0.
    cfg.CONT.DIST.MASK_KD = 0.
    # cfg.CONT.DIST.SANITY = 1.
    # cfg.CONT.DIST.WEIGHT_MASK = 1.
    
    # Parameters for ECLIPSE
    cfg.CONT.NUM_PROMPTS = 10
    cfg.CONT.SOFTCLS = True
    cfg.CONT.BACKBONE_FREEZE = False
    cfg.CONT.CLS_HEAD_FREEZE = False
    cfg.CONT.MASK_HEAD_FREEZE = False
    cfg.CONT.PIXEL_DECODER_FREEZE = False
    cfg.CONT.QUERY_EMBED_FREEZE = False
    cfg.CONT.TRANS_DECODER_FREEZE = False
    cfg.CONT.PROMPT_DEEP = False
    cfg.CONT.PROMPT_MASK_MLP = False
    cfg.CONT.PROMPT_NO_OBJ_MLP = False
    cfg.CONT.DEEP_CLS = False
    cfg.CONT.LOGIT_MANI_DELTAS = None
    
