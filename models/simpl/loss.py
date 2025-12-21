from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunc(nn.Module):
    def __init__(self, cfg):
        super(LossFunc, self).__init__()
        self.config = cfg

        self.yaw_loss = cfg.yaw_loss
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, data):
        traj_fut = data["TRAJS_POS_FUT"] # [x["TRAJS_POS_FUT"] for x in data["TRAJS"]]
        traj_fut = traj_fut

        pad_fut = data["PAD_FUT"] #[x["PAD_FUT"] for x in data["TRAJS"]]
        pad_fut = [i.long() for i in pad_fut]

        cls, reg, aux = out


        train_mask = data["TRAIN_MASK"] # [x["TRAIN_MASK"] for x in data["TRAJS"]]
        train_mask = train_mask

        cls = [x[train_mask[i]] for i, x in enumerate(cls)]
        reg = [x[train_mask[i]] for i, x in enumerate(reg)]
        traj_fut = [x[train_mask[i]] for i, x in enumerate(traj_fut)]
        pad_fut = [x[train_mask[i]] for i, x in enumerate(pad_fut)]


        if self.yaw_loss:
            # yaw angle GT
            ang_fut = data["TRAJS_ANG_FUT"]# [x["TRAJS_ANG_FUT"] for x in data["TRAJS"]]
            ang_fut = ang_fut
            # for yaw loss
            yaw_loss_mask = data["YAW_LOSS_MASK"] #[x["YAW_LOSS_MASK"] for x in data["TRAJS"]]

            # collect aux info
            vel = [x[0] for x in aux]
            # apply train mask
            vel = [x[train_mask[i]] for i, x in enumerate(vel)]
            ang_fut = [x[train_mask[i]] for i, x in enumerate(ang_fut)]
            yaw_loss_mask = [x[train_mask[i]] for i, x in enumerate(yaw_loss_mask)]

            loss_out = self.pred_loss_with_yaw(
                cls, reg, vel, traj_fut, ang_fut, pad_fut, yaw_loss_mask
            )
            loss_out["loss"] = (
                loss_out["cls_loss"] + loss_out["reg_loss"] + loss_out["yaw_loss"]
            )
        else:
            loss_out = self.pred_loss(cls, reg, traj_fut, pad_fut)
            loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"]

        return loss_out

    def pred_loss_with_yaw(
        self,
        cls: List[torch.Tensor],
        reg: List[torch.Tensor],
        vel: List[torch.Tensor],
        gt_preds: List[torch.Tensor],
        gt_ang: List[torch.Tensor],
        pad_flags: List[torch.Tensor],
        yaw_flags: List[torch.Tensor],
    ):
        cls = torch.cat([x for x in cls], dim=0)  # [98, 6]
        reg = torch.cat([x for x in reg], dim=0)  # [98, 6, 60, 2]
        vel = torch.cat([x for x in vel], dim=0)  # [98, 6, 60, 2]
        gt_preds = torch.cat([x for x in gt_preds], dim=0)  # [98, 60, 2]
        gt_ang = torch.cat([x for x in gt_ang], dim=0)  # [98, 60, 2]
        has_preds = torch.cat([x for x in pad_flags], dim=0).bool()  # [98, 60]
        has_yaw = torch.cat([x for x in yaw_flags], dim=0).bool()  # [98]

        loss_out = dict()
        num_modes = self.config.k
        num_preds = self.config.global_pred_lane
        # assert(has_preds.all())

        ar = torch.arange(num_preds, device=has_preds.device, dtype=torch.float32)
        last = has_preds.float() + 0.1 * ar / float(num_preds)

        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        vel = vel[mask]
        gt_preds = gt_preds[mask]
        gt_ang = gt_ang[mask]
        has_preds = has_preds[mask]
        has_yaw = has_yaw[mask]
        last_idcs = last_idcs[mask]

        _reg = reg[..., 0:2].clone()  # for WTA strategy

        row_idcs = torch.arange(len(last_idcs)).long()
        dist = []
        for j in range(num_modes):
            dist.append(
                torch.sqrt(
                    (
                        (_reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long()

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config.cls_thres).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config.cls_ignore
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config.mgn
        num_cls = mask.sum().item()
        cls_loss = (self.config.mgn * mask.sum() - mgn[mask].sum()) / (
            num_cls + 1e-10
        )
        loss_out["cls_loss"] = self.config.cls_coef * cls_loss

        reg = reg[row_idcs, min_idcs]
        num_reg = has_preds.sum().item()
        reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (
            num_reg + 1e-10
        )
        loss_out["reg_loss"] = self.config.reg_coef * reg_loss

        # ~ yaw loss
        vel = vel[row_idcs, min_idcs]  # select the best mode, keep identical to reg

        _has_preds = has_preds[has_yaw].view(-1)
        _v1 = vel[has_yaw].view(-1, 2)[_has_preds]
        _v2 = gt_ang[has_yaw].view(-1, 2)[_has_preds]
        # ang diff loss use cosine similarity
        cos_sim = torch.cosine_similarity(_v1, _v2)  # [-1, 1]
        loss_out["yaw_loss"] = ((1 - cos_sim) / 2).mean()  # [0, 1]

        return loss_out

    def pred_loss(
        self,
        cls: List[torch.Tensor],
        reg: List[torch.Tensor],
        gt_preds: List[torch.Tensor],
        pad_flags: List[torch.Tensor],
    ):
        cls = torch.cat([x for x in cls], 0)  # [98, 6]
        reg = torch.cat([x for x in reg], 0)  # [98, 6, 60, 2]
        gt_preds = torch.cat([x for x in gt_preds], 0)  # [98, 60, 2]
        has_preds = torch.cat([x for x in pad_flags], 0).bool()  # [98, 60]

        loss_out = dict()
        num_modes = self.config.k
        num_preds = self.config.global_pred_lane
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float() / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        _reg = reg[..., 0:2].clone()  # for WTA strategy

        row_idcs = torch.arange(len(last_idcs)).long()
        dist = []
        for j in range(num_modes):
            dist.append(
                torch.sqrt(
                    (
                        (_reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long()

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config.cls_thres).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config.cls_ignore
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config.mgn
        num_cls = mask.sum().item()
        cls_loss = (self.config.mgn * mask.sum() - mgn[mask].sum()) / (
            num_cls + 1e-10
        )
        loss_out["cls_loss"] = self.config.cls_coef* cls_loss

        reg = reg[row_idcs, min_idcs]
        num_reg = has_preds.sum().item()
        reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (
            num_reg + 1e-10
        )
        loss_out["reg_loss"] = self.config.reg_coef * reg_loss

        return loss_out
