# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

ADE20K_SEM_SEG_CATEGORIES = [
    "void", "wall", "building", "sky", "floor", "tree", "ceiling", "road, route", "bed", "window ", "grass", "cabinet", "sidewalk, pavement", "person", "earth, ground", "door", "table", "mountain, mount", "plant", "curtain", "chair", "car", "water", "painting, picture", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "tub", "rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove", "palm, palm tree", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus", "towel", "light", "truck", "tower", "chandelier", "awning, sunshade, sunblind", "street lamp", "booth", "tv", "plane", "dirt track", "clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "pool", "stool", "barrel, cask", "basket, handbasket", "falls", "tent", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light", "tray", "trash can", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass, drinking glass", "clock", "flag", # noqa
]

def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations", dirname)
        name = f"myade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k(_root)