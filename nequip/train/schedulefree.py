from .lightning import NequIPLightningModule
from typing import Dict, Any
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class ScheduleFreeLightningModule(NequIPLightningModule):
    """
    NequIP LightningModule using Facebook's Schedule-Free optimizer.

    This module wraps the model's optimizer in one of Facebook's Schedule-Free variants.
    See: https://github.com/facebookresearch/schedule_free

    Args:
        optimizer (Dict[str, Any]): Dictionary that must include a `_target_`
            corresponding to one of the Schedule-Free optimizers and other keyword arguments
            compatible with the Schedule-Free variants.
    """

    def __init__(self, optimizer: Dict[str, Any], **kwargs):
        valid_targets = {
            "AdamWScheduleFree",
            "SGDScheduleFree",
            "RAdamScheduleFree",
        }
        if "_target_" not in optimizer or not any(
            optimizer["_target_"].endswith(name) for name in valid_targets
        ):
            raise MisconfigurationException(
                f"Invalid optimizer: expected Schedule-Free optimizer (_target_ ending with one of {valid_targets}), "
            )

        self.schedulefree_optimizer_class = optimizer["_target_"]
        super().__init__(optimizer=optimizer, **kwargs)

    def on_train_epoch_start(self):
        for opt in self.trainer.optimizers:
            if hasattr(opt, "train"):
                opt.train()

    def on_validation_epoch_start(self):
        for opt in self.trainer.optimizers:
            if hasattr(opt, "eval"):
                opt.eval()

    def on_test_epoch_start(self):
        for opt in self.trainer.optimizers:
            if hasattr(opt, "eval"):
                opt.eval()

    def on_predict_epoch_start(self):
        for opt in self.trainer.optimizers:
            if hasattr(opt, "eval"):
                opt.eval()
