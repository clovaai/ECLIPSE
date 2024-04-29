from fvcore.common.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from detectron2.engine.train_loop import HookBase
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
from detectron2.engine.hooks import EvalHook
from detectron2.evaluation.testing import flatten_results_dict
import detectron2.utils.comm as comm


class BetterEvalHook(EvalHook):
    def _do_eval(self):
        results = self._func()
        if "confusion_matrix" in results:
            cf_mtx = results.pop("confusion_matrix")
            self.trainer.storage.put_image("confusion_matrix", cf_mtx)

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()


class BetterPeriodicCheckpointer(_PeriodicCheckpointer, HookBase):
    """
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        # No way to use **kwargs
        self.step(self.trainer.iter)

    def step(self, iteration: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) % self.period == 0:
            # this is the only thing changed. I want only the last model, not many intermediate
            self.checkpointer.save(
                f"{self.file_prefix}_tmp", **additional_state
            )

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        self.path_manager.rm(file_to_delete)

        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)
