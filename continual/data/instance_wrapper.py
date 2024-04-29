from typing import Callable, List, Union, Optional, Callable
import os
import multiprocessing
from functools import partial

import numpy as np

from detectron2.data import build_detection_train_loader, build_detection_test_loader

from continuum.scenarios import ClassIncremental
from continuum.tasks import TaskSet
from continuum.download import ProgressBar

from PIL import Image
import torch
import logging
import torchvision


class InstanceContinualDetectron(ClassIncremental):

    def __init__(
        self,
        cl_dataset: List[dict],
        nb_classes: int,
        increment: Union[List[int], int] = 0,
        initial_increment: int = 0,
        mapper: Callable = None,
        class_order: Optional[List[int]] = None,
        mode: str = "overlap",
        save_indexes: Optional[str] = None,
        masking_value: Optional[int] = None,  # NEVER USED
        cfg = None
    ) -> None:
        self.mode = mode
        self.save_indexes = save_indexes + ".npy"
        self._nb_classes = nb_classes
        self.mapper = mapper
        self.cfg = cfg

        if self.mode not in ("overlap", "disjoint", "sequential"):
            raise ValueError(f"Unknown mode={mode}.")

        if class_order is not None:
            if len(class_order) != nb_classes:
                raise ValueError(
                    f"Number of classes ({nb_classes}) != class ordering size ({len(class_order)}."
                )

        self.cl_dataset = cl_dataset
        self.increment = increment
        self.initial_increment = initial_increment
        self.class_order = class_order

        self._nb_tasks = self._setup(None)

    @property
    def nb_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        return self._nb_classes

    def _get_label_transformation(self, task_index: Union[int, List[int]]):
        """Returns the transformation to apply on the GT segmentation maps.

        :param task_index: The selected task id.
        :return: A pytorch transformation.
        """
        if isinstance(task_index, int):
            task_index = [task_index]
        if not self.train:
            # In testing mode, all labels brought by previous tasks are revealed
            task_index = list(range(max(task_index) + 1))

        if self.mode in ("overlap", "disjoint"):
            # Previous and future (for disjoint) classes are hidden
            labels = self._get_task_labels(task_index)
        elif self.mode == "sequential":
            # Previous classes are not hidden, no future classes are present
            labels = self._get_task_labels(list(range(max(task_index) + 1)))
        else:
            raise ValueError(f"Unknown mode={self.mode}.")

        inverted_order = {label: self.class_order.index(label) for label in labels}
        inverted_order[255] = 255

        return torchvision.transforms.Lambda(
            lambda seg_map: seg_map.apply_(
                lambda v: inverted_order.get(v, 255)
            )
        )

    def _get_task_labels(self, task_indexes: Union[int, List[int]]) -> List[int]:
        """Returns the labels/classes of the current tasks, not all present labels!

        :param task_indexes: The selected task ids.
        :return: A list of class/labels ids.
        """
        if isinstance(task_indexes, int):
            task_indexes = [task_indexes]

        labels = set()
        for t in task_indexes:
           previous_inc = sum(self._increments[:t])
           labels.update(
               self.class_order[previous_inc:previous_inc+self._increments[t]]
           )

        return list(labels)

    def _get_task_ids(self, t: np.ndarray, task_indexes: Union[int, List[int]]) -> np.ndarray:
        """Reduce multiple task ids to a single one per sample.

        In segmentation, the same image can have several task ids. We assume that
        the selected task ids is the last one as it will be the one that matter
        for the ground-truth segmentation maps.

        :param t: A matrix of task ids of shape (nb_samples, nb_tasks).
        :param task_indexes: The selected task ids.
        :return: An array of task ids of shape (nb_samples,).
        """
        if isinstance(task_indexes, list):
            task_indexes = max(task_indexes)
        return np.ones((len(t))) * task_indexes

    def __getitem__(self, task_index: Union[int, slice]) -> TaskSet:
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice) and task_index.step is not None:
            raise ValueError("Step in slice for segmentation is not supported.")

        logger = logging.getLogger(__name__)

        y, t, task_index, _ = self._select_data_by_task(task_index)
        t = self._get_task_ids(t, task_index)

        target_trsf = self._get_label_transformation(task_index)

        def mapper_wrapper(dataset_dict):
            dataset_dict = self.mapper(dataset_dict) if self.mapper is not None else dataset_dict
            if "instances" in dataset_dict:
                """ instances istype Instances (from detectron2.structures.instances) """
                instances = dataset_dict["instances"]
                instances.gt_classes = target_trsf(instances.gt_classes)

                not_ignore = (instances.gt_classes != 255)

                new_gt_classes = instances.gt_classes[not_ignore]
                new_gt_masks = instances.gt_masks[not_ignore]

                # Work around to reduce the number of instances (modifying in place would make it crash)
                instances.remove('gt_classes')
                instances.remove('gt_masks')
                if hasattr(instances, "gt_boxes"):
                    new_gt_boxes = instances.gt_boxes[not_ignore]
                    instances.remove('gt_boxes')
                    instances.gt_boxes = new_gt_boxes
                instances.gt_classes = new_gt_classes
                instances.gt_masks = new_gt_masks

            return dataset_dict

        if self.train:
            logger.info(f"There are {len(y)} images in the train dataset.")
            return build_detection_train_loader(
                self.cfg,
                mapper=mapper_wrapper,
                dataset=y,
            )
        else:
            logger.info(f"There are {len(y)} images in the test dataset.")
            return build_detection_test_loader(
                mapper=mapper_wrapper,
                dataset=y,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            )

    def get_original_targets(self, targets: np.ndarray) -> np.ndarray:
        """Returns the original targets not changed by the custom class order.

        :param targets: An array of targets, as provided by the task datasets.
        :return: An array of targets, with their original values.
        """
        return self._class_mapping(targets)

    def _select_data_by_task(
            self,
            task_index: Union[int, slice, np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, Union[int, List[int]]]:
        """Selects a subset of the whole data for a given task.
        This class returns the "task_index" in addition of the x, y, t data.
        This task index is either an integer or a list of integer when the user
        used a slice. We need this variable when in segmentation to disentangle
        samples with multiple task ids.
        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A tuple of numpy array being resp. (1) the data, (2) the targets,
                 (3) task ids, and (4) the actual task required by the user.
        """

        # conversion of task_index into a list

        if isinstance(task_index, slice):
            start = task_index.start if task_index.start is not None else 0
            stop = task_index.stop if task_index.stop is not None else len(self) + 1
            step = task_index.step if task_index.step is not None else 1
            task_index = list(range(start, stop, step))
            if len(task_index) == 0:
                raise ValueError(f"Invalid slicing resulting in no data (start={start}, end={stop}, step={step}).")

        if isinstance(task_index, np.ndarray):
            task_index = list(task_index)

        y, t = self.dataset  # type: ignore

        if isinstance(task_index, list):
            task_index = [
                t if t >= 0 else _handle_negative_indexes(t, len(self)) for t in task_index
            ]
            if len(t.shape) == 2:
                data_indexes = np.unique(np.where(t[:, task_index] == 1)[0])
            else:
                data_indexes = np.where(np.isin(t, task_index))[0]
        else:
            if task_index < 0:
                task_index = _handle_negative_indexes(task_index, len(self))

            if len(t.shape) == 2:
                data_indexes = np.where(t[:, task_index] == 1)[0]
            else:
                data_indexes = np.where(t == task_index)[0]

        selected_y = [y[index] for index in data_indexes]
        selected_t = t[data_indexes]

        return selected_y, selected_t, task_index, data_indexes

    @property
    def train(self) -> bool:
        """Returns whether we are in training or testing mode.
        This property is dependent on the dataset/mapper, not the actual scenario.
        """
        return self.mapper.is_train if self.mapper is not None else True

    def _setup(self, nb_tasks: int) -> int:
        """Setups the different tasks."""
        y = self.cl_dataset

        self.class_order = self.class_order or list(
            range(0, self._nb_classes))

        # For when the class ordering is changed,
        # so we can quickly find the original labels
        def class_mapping(c):
            return self.class_order[c]
        self._class_mapping = np.vectorize(class_mapping)

        self._increments = self._define_increments(
            self.increment, self.initial_increment, self.class_order
        )

        # Checkpointing the indexes if the option is enabled.
        # The filtering can take multiple minutes, thus saving/loading them can
        # be useful.
        if self.train:
            if self.save_indexes is not None and os.path.exists(self.save_indexes):
                print(f"Loading previously saved indexes ({self.save_indexes}).")
                t = np.load(self.save_indexes)
            else:
                print("Computing indexes, it may be slow!")
                t = _filter_images(
                    y, self._increments, self.class_order, self.mode
                )
                if self.save_indexes is not None:
                    np.save(self.save_indexes, t)
        else:
            t = np.ones((len(y), len(self._increments)))

        assert len(y) == len(t) and len(t) > 0

        self.dataset = (y, t)

        return len(self._increments)


def _filter_images(
    paths: Union[np.ndarray, List[str]],
    increments: List[int],
    class_order: List[int],
    mode: str = "overlap"
) -> np.ndarray:
    """Select images corresponding to the labels.

    Strongly inspired from Cermelli's code:
    https://github.com/fcdl94/MiB/blob/master/dataset/utils.py#L19

    :param paths: An iterable of paths to gt maps.
    :param increments: All individual increments.
    :param class_order: The class ordering, which may not be [1, 2, ...]. The
                        background class (0) and unknown class (255) aren't
                        in this class order.
    :param mode: Mode of the segmentation (see scenario doc).
    :return: A binary matrix representing the task ids of shape (nb_samples, nb_tasks).
    """
    indexes_to_classes = []
    pb = ProgressBar()

    find_classes = partial(_find_classes)

    with multiprocessing.Pool(min(8, multiprocessing.cpu_count())) as pool:
        for i, classes in enumerate(pool.imap(find_classes, paths), start=1):
            indexes_to_classes.append(classes)
            if i % 100 == 0:
                pb.update(None, 100, len(paths))
        pb.end(len(paths))

    t = np.zeros((len(paths), len(increments)))
    accumulated_inc = 0

    for task_id, inc in enumerate(increments):
        labels = class_order[accumulated_inc:accumulated_inc+inc]
        old_labels = class_order[:accumulated_inc]
        all_labels = labels + old_labels + [255]

        for index, classes in enumerate(indexes_to_classes):
            if mode == "overlap":
                if any(c in labels for c in classes):
                    t[index, task_id] = 1
            elif mode in ("disjoint", "sequential"):
                if any(c in labels for c in classes) and all(c in all_labels for c in classes):
                    t[index, task_id] = 1
            else:
                raise ValueError(f"Unknown mode={mode}.")

        accumulated_inc += inc

    return t


def _find_classes(image_dict: dict, class_mapper: Optional[Callable] = None) -> set:
    """Open a ground-truth segmentation map image and returns all unique classes
    contained.

    :param path: Path to the image.
    :return: Unique classes.
    """
    if "segments_info" in image_dict:  # Panoptic segmentation case
        if class_mapper:
            return set(class_mapper(segment["category_id"]) for segment in image_dict["segments_info"])
        return set(segment["category_id"] for segment in image_dict["segments_info"])
    elif 'sem_seg_file_name' in image_dict:  # Semantic segmentation case
        path = image_dict['sem_seg_file_name']
        if class_mapper:
            return set(np.unique(np.array(Image.open(path)).reshape(-1)))
        return np.unique(np.array(Image.open(path)).reshape(-1))
    elif 'annotations' in image_dict:  # Instance segmentation case
        if class_mapper:
            return set(class_mapper(segment["category_id"]) for segment in image_dict["annotations"])
        return set(segment["category_id"] for segment in image_dict["annotations"])
    else:
        raise NotImplementedError("Unable to infer the correct task in ContinualDetectron.")


def _handle_negative_indexes(index: int, total_len: int) -> int:
    if index < 0:
        index = index % total_len
    return index