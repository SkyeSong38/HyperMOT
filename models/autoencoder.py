import torch
from torch.nn import Module
import models.diffusion as diffusion
from models.diffusion import VarianceSchedule, D2MP_OB
from models.motion_decoder import Time_info_decoder
import numpy as np


class D2MP(Module):
    def __init__(self, config, encoder=None, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = encoder
        if config.use_diffmot:
            self.diffnet = getattr(diffusion, config.diffnet)
            self.diffusion = D2MP_OB(
                # net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
                net=self.diffnet(point_dim=4, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
                var_sched=VarianceSchedule(
                    num_steps=100,
                    beta_T=5e-2,
                    mode='linear'
                ),
                config=self.config
            )
        else:
            self.ssm_decoder = Time_info_decoder()

    @torch.no_grad()
    def generate(self, conds, sample, bestof, flexibility=0.0, ret_traj=False, img_w=None, img_h=None):
        if not conds:
            return np.empty((0, 4), dtype=np.float32)
        cond_encodeds = []
        for i in range(len(conds)):
            tmp_c = conds[i]
            tmp_c = np.array(tmp_c, dtype=np.float32)[-self.config.interval:]
            tmp_c[:, 0::2] = tmp_c[:, 0::2] / img_w
            tmp_c[:, 1::2] = tmp_c[:, 1::2] / img_h
            tmp_conds = torch.tensor(tmp_c, dtype=torch.float)
            if len(tmp_conds) != self.config.interval:
                pad_conds = tmp_conds[-1].repeat((self.config.interval, 1))
                tmp_conds = torch.cat((tmp_conds, pad_conds), dim=0)[:self.config.interval]
            cond_encodeds.append(tmp_conds.unsqueeze(0))
        cond_encodeds = torch.cat(cond_encodeds).to(next(self.parameters()).device)
        if self.config.use_diffmot:
            cond_flow = self.encoder(cond_encodeds)
            track_pred = self.diffusion.sample(cond_flow, sample, bestof, flexibility=flexibility, ret_traj=ret_traj)
        else:
            active_hyper = self.config.active_hyper
            if active_hyper:
                # All supplied tracks belong to the same video frame.
                cond_encodeds = cond_encodeds.unsqueeze(0)
            cond_flow = self.encoder(cond_encodeds, decoder_only=self.config.decoder_only,
                                     active_hyper=active_hyper)
            positions = cond_encodeds[..., :4]
            if self.config.one2one_predict:
                positions = positions[..., -1:, :]
                cond_flow = cond_flow[..., -1:, :]
            valid_mask = torch.ones(cond_encodeds.shape[:2], dtype=torch.bool,
                                    device=cond_encodeds.device) if active_hyper else None
            track_pred = self.ssm_decoder(positions, cond_flow, active_hyper=active_hyper,
                                          valid_mask=valid_mask)
            track_pred = track_pred[..., -1, :]
            if active_hyper:
                track_pred = track_pred.squeeze(0)
        return track_pred.cpu().detach().numpy()

    def forward(self, batch):
        # config["one2one_predict"] = False
        # config["decoder_only"] = True
        # config["AR_data"] = True
        # config["hyper_prediction"] = True

        valid_mask = None
        if self.config.active_hyper:
            condition = batch['condition'].permute(0, 2, 1, 3)
            next_bbox = batch['gt_bbox'].permute(0, 2, 1, 3)
            valid_mask = batch['valid_mask'].to(device=condition.device, dtype=torch.bool)
            condition = condition.masked_fill(~valid_mask[:, :, None, None], 0)
            if self.config.one2one_predict:
                cur_batch = condition[:, :, -1:, :4]
                cur_gt = next_bbox
            else:
                cur_batch = condition[..., :4]
                cur_gt = torch.cat((condition[:, :, 1:, :4], next_bbox), dim=2)
        else:
            condition = batch['condition']
            if self.config.one2one_predict:
                cur_batch = batch["condition"][:, -1, :4].unsqueeze(1)
                cur_gt = batch["cur_bbox"].unsqueeze(1)
            else:
                cur_batch = batch["condition"][:, :, :4]
                cur_gt = torch.cat((batch["condition"][:, 1:, :4], batch["cur_bbox"].unsqueeze(1)), dim=1)

        cond_encoded = self.encoder(condition,
                                    decoder_only=self.config.decoder_only,
                                    active_hyper=self.config.active_hyper)  # input： batch["condition"] b 5 8。 out： b 1 256

        if self.config.one2one_predict:
            cond_encoded = cond_encoded[..., -1:, :]
        loss = self.ssm_decoder(cur_batch, cond_encoded, curr_gt=cur_gt,
                                active_hyper=self.config.active_hyper, valid_mask=valid_mask)
        return loss
