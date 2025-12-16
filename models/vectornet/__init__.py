import torch
import pytorch_lightning as pl

from .vectornet import VectorNetTrajPred


class VectorNetLightningModule(pl.LightningModule):
    def __init__(self, lr, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = VectorNetTrajPred(*args, **kwargs)

    def forward(self, batch: dict) -> torch.Tensor:

        return self.model(batch)

    def training_step(self, batch, batch_idx):
        # pred, logits = self.model(batch) # TODO
        pred = self.model(batch)
        loss = self.model.loss(pred, batch)

        losses = {
            "loss": loss,
        }

        self._log_losses(
            losses, "train", batch_size=batch["target_gt"].shape[0])
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        # pred, logits = self.model(batch) # TODO
        pred = self.model(batch)
        loss = self.model.loss(pred, batch)
        losses = {
            "loss": loss,
        }
        self._log_losses(losses, "val", batch_size=batch["target_gt"].shape[0])

    def test_step(self, batch: dict, batch_idx: int):
        # pred, logits = self.model(batch) # TODO
        pred = self.model(batch)
        loss = self.model.loss(pred, batch)
        losses = {
            "loss": loss,
        }
        self._log_losses(
            losses, "test", batch_size=batch["target_gt"].shape[0])

    def _log_losses(self, loss_dict, prefix: str, batch_size: int):
        self.log(
            f"{prefix}/loss",
            loss_dict["loss"],
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
