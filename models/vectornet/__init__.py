import torch
import pytorch_lightning as pl

from .vectornet import VectorNetTrajPred

from ..metrics import ADE, FDE, minADE, minFDE


class VectorNetLightningModule(pl.LightningModule):
    def __init__(self, lr, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = VectorNetTrajPred(*args, **kwargs)

    def forward(self, batch: dict) -> torch.Tensor:

        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pred, logits = self.model(batch)
        losses = self.model.loss(pred, logits, batch)

        self._log_losses(
            losses, "train", batch_size=batch["target_gt"].shape[0])
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        pred, logits = self.model(batch)
        losses = self.model.loss(pred, logits, batch)

        self._log_losses(losses, "val", batch_size=batch["target_gt"].shape[0])

        if self.model.k == 1:
            # single modal metrics ade, fde
            ade = ADE(pred, batch["target_gt"]).mean()
            fde = FDE(pred, batch["target_gt"]).mean()
            self.log("val/ADE", ade, prog_bar=True, on_epoch=True,
                     batch_size=batch["target_gt"].shape[0])
            self.log("val/FDE", fde, prog_bar=True, on_epoch=True,
                     batch_size=batch["target_gt"].shape[0])
        else:
            # multi modal metrics minade, minfde
            min_ade = minADE(pred, batch["target_gt"]).mean()
            min_fde = minFDE(pred, batch["target_gt"]).mean()
            self.log("val/minADE", min_ade, prog_bar=True,
                     on_epoch=True, batch_size=batch["target_gt"].shape[0])
            self.log("val/minFDE", min_fde, prog_bar=True,
                     on_epoch=True, batch_size=batch["target_gt"].shape[0])

    def test_step(self, batch: dict, batch_idx: int):
        pred, logits = self.model(batch)
        losses = self.model.loss(pred, logits, batch)

        self._log_losses(
            losses, "test", batch_size=batch["target_gt"].shape[0])

    def _log_losses(self, loss_dict, prefix: str, batch_size: int):
        for k, v in loss_dict.items():
            self.log(
                f"{prefix}/{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                on_step=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
