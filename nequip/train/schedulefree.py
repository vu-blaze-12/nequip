import torch
from .lightning import NequIPLightningModule
from schedulefree import AdamWScheduleFree
from nequip.data import AtomicDataDict
from typing import Optional, Dict


class ScheduleFreeLightningModule(NequIPLightningModule):
    """
    NequIP LightningModule using Facebook's Schedule-Free optimizer.

    This module wraps the model's optimizer in AdamWScheduleFree to enable
    schedule-free training. See: https://github.com/facebookresearch/schedule_free

    Args:
        optimizer (Dict): Dictionary of keyword arguments compatible with
            AdamWScheduleFree. See: https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py
    """

    def __init__(
        self,
        optimizer: Dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.automatic_optimization = True
        self.optimizer_params = optimizer
        self._sf_optimizer = None

    def configure_optimizers(self):
        # Extract trainable params
        param_groups = [p for p in self.model.parameters() if p.requires_grad]
        opt_args = {
            k: v for k, v in self.optimizer_params.items() if not k.startswith("_")
        }

        # Initialize Schedule-Free optimizer
        self._sf_optimizer = AdamWScheduleFree(
            params=param_groups,
            **opt_args,
        )
        return self._sf_optimizer

    def on_train_epoch_start(self):
        if self._sf_optimizer is not None:
            self._sf_optimizer.train()

    def on_validation_epoch_start(self):
        if self._sf_optimizer is not None:
            self._sf_optimizer.eval()

    def on_test_epoch_start(self):
        if self._sf_optimizer is not None:
            self._sf_optimizer.eval()

    def on_predict_epoch_start(self):
        if self._sf_optimizer is not None:
            self._sf_optimizer.eval()
