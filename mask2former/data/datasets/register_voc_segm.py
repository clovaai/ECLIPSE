import os
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog

classes = [
    (0,  'background'),
    (1, 'aeroplane'),
    (2, 'bicycle'),
    (3, 'bird'),
    (4, 'boat'),
    (5, 'bottle'),
    (6, 'bus'),
    (7, 'car'),
    (8, 'cat'),
    (9, 'chair'),
    (10, 'cow'),
    (11, 'diningtable'),
    (12, 'dog'),
    (13, 'horse'),
    (14, 'motorbike'),
    (15, 'person'),
    (16, 'pottedplant'),
    (17, 'sheep'),
    (18, 'sofa'),
    (19, 'train'),
    (20, 'tvmonitor')
]


def get_voc_data(root, train=True, base_dir='PascalVOC2012'):

        root = os.path.expanduser(root)

        voc_root = os.path.join(root, base_dir)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted. Download it and put in the dataset folder.')

        mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
        assert os.path.exists(mask_dir), "SegmentationClassAug not found"

        if train:
            split_f = os.path.join(splits_dir, 'train_aug.txt')
        else:
            split_f = os.path.join(splits_dir, 'val.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]

        # create the dict
        # then return it.
        dataset_dicts = []
        for (img_path, gt_path) in images:
            record = {}
            record["file_name"] = img_path
            record["sem_seg_file_name"] = gt_path
            dataset_dicts.append(record)
        return dataset_dicts


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def _get_voc_meta():
    colors = list(voc_cmap(21))
    return {"stuff_classes": [k[1] for k in classes], "stuff_colors": colors}


def register_voc(root):
    meta = _get_voc_meta()
    for name, train in [("voc_segm_train", True), ("voc_segm_val", False)]:
        DatasetCatalog.register(
            name, lambda r=root, t=train: get_voc_data(r, train=t)
        )
        MetadataCatalog.get(name).set(
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_voc(_root)
