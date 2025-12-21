import torch
import pytorch_lightning as pl

from .simpl import Simpl

from ..metrics import ADE, FDE, minADE, minFDE


class SimplLightningModule(pl.LightningModule):
    def __init__(self, lr, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = Simpl(*args, **kwargs)
        self.k = self.model.pred_net.k

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch, batch_idx):

        num_agents = batch["agent_feats"].shape[1]
        num_lanes = batch["lane_feats"].shape[1]

        self.log(
            "train/num_agents",
            num_agents,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/num_lanes",
            num_lanes,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            "train/num_all",
            num_agents + num_lanes,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

        out = self.model(batch)  # out = (res_cls, res_reg, res_aux)
        post_out = self.model.post_process(out)
        pred = post_out["traj_pred"]
        logits = post_out["prob_pred"]

        losses = self.model.loss(out, batch)

        self._log_losses(losses, "train", batch_size=pred.shape[0])
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        out = self.model(batch)  # out = (res_cls, res_reg, res_aux)
        post_out = self.model.post_process(out)
        pred = post_out["traj_pred"]
        logits = post_out["prob_pred"]

        losses = self.model.loss(out, batch)

        self._log_losses(losses, "val", batch_size=pred.shape[0])

        # if self.model.k == 1:
        #     # single modal metrics ade, fde
        #     ade = ADE(pred, batch["target_gt"]).mean()
        #     fde = FDE(pred, batch["target_gt"]).mean()
        #     self.log("val/ADE", ade, prog_bar=True, on_epoch=True,
        #              batch_size=batch["target_gt"].shape[0])
        #     self.log("val/FDE", fde, prog_bar=True, on_epoch=True,
        #              batch_size=batch["target_gt"].shape[0])
        # else:
        #     # multi modal metrics minade, minfde
        #     min_ade = minADE(pred, batch["target_gt"]).mean()
        #     min_fde = minFDE(pred, batch["target_gt"]).mean()
        #     self.log("val/minADE", min_ade, prog_bar=True,
        #              on_epoch=True, batch_size=batch["target_gt"].shape[0])
        #     self.log("val/minFDE", min_fde, prog_bar=True,
        #              on_epoch=True, batch_size=batch["target_gt"].shape[0])

    def test_step(self, batch: dict, batch_idx: int):
        res_cls, res_reg, res_aux = self.model(batch)
        out = self.model.post_process((res_cls, res_reg, res_aux))
        pred = out["traj_pred"]
        logits = out["prob_pred"]

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

    def run_forward_postprocess(self, batch: dict) -> torch.Tensor:
        out = self.model(batch)
        out_post = self.model.post_process(out)

        pred = out_post["traj_pred"]
        prob = out_post["prob_pred"]

        return pred, prob
    
    def create_scenario(self, batch, outputs, index: int = 0):
        """
        batch: dict
        agent_feats: (b, n_agent_max, dim, t)
        agent_masks: (b, n_agent_max)
        lane_feats: (b, n_lane_max, dim, t)
        lane_masks: (b, n_lane_max)
        "TRAJS_FUT": (b, n_agent_max, future_len, 2)
        "PAD_FUT": (b, n_agent_max, future_len)
        
        return dict:
        lane_points: tensor, (num_lanes, num_points, 2)
        agent_history: tensor, (num_agents, hist_len, feat_dim)
        target_agent_idx: int
        target_last_pos: tensor, (2,)
        target_gt: tensor, (future_len, 2)
        prediction: tensor, (k, future_len, 2)
        other_future: tensor, (num_agents, future_len, 2) or None
        other_future_mask: tensor, (num_agents, future_len) or None
        other_prediction: tensor, (num_agents, k, future_len, 2) or None
        agent_last_pos: tensor, (num_agents, 2) or None
        """
        agent_inputs = batch["agent_feats"][index]  # (num_agents, hist_len, feat_dim)
