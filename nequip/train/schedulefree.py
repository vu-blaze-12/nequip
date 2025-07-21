import torch
from .lightning import NequIPLightningModule
from schedulefree import AdamWScheduleFree  # or SGDScheduleFree, etc.
from nequip.data import AtomicDataDict
from typing import Optional, Dict


class ScheduleFreeLightningModule(NequIPLightningModule):
    def __init__(
        self,
        sf_lr: float = 1e-3,
        sf_weight_decay: float = 1e-4,
        sf_momentum: float = 0.9,
        sf_warmup_steps: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.automatic_optimization = False
        self.sf_lr = sf_lr
        self.sf_weight_decay = sf_weight_decay
        self.sf_momentum = sf_momentum
        self.sf_warmup_steps = sf_warmup_steps
        self._sf_optimizer = None

    def configure_optimizers(self):
        param_groups = [p for p in self.model.parameters() if p.requires_grad]
        self._sf_optimizer = AdamWScheduleFree(
            params=param_groups,
            lr=self.sf_lr,
            weight_decay=self.sf_weight_decay,
            warmup_steps=self.sf_warmup_steps,
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

    def training_step(self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0):
        target = self.process_target(batch, batch_idx, dataloader_idx)
        output = self(batch)

        if self.train_metrics is not None:
            with torch.no_grad():
                metric_dict = self.train_metrics(
                    output, target, prefix=f"train_metric_step{self.logging_delimiter}"
                )
            self.log_dict(metric_dict)

        loss_dict = self.loss(
            output, target, prefix=f"train_loss_step{self.logging_delimiter}"
        )
        self.log_dict(loss_dict)
        loss = (
            loss_dict[f"train_loss_step{self.logging_delimiter}weighted_sum"]
            * self.world_size
        )

        self.manual_backward(loss)
        self._sf_optimizer.step()
        self._sf_optimizer.zero_grad(set_to_none=True)

        return loss

    def on_train_epoch_end(self):
        if self.train_metrics is not None:
            train_metric_dict = self.train_metrics.compute(
                prefix=f"train_metric_epoch{self.logging_delimiter}"
            )
            self.log_dict(train_metric_dict)
            self.train_metrics.reset()

        loss_dict = self.loss.compute(prefix=f"train_loss_epoch{self.logging_delimiter}")
        self.log_dict(loss_dict)
        self.loss.reset()
