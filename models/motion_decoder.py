import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss
from .box_ops import box_cxcywh_to_xyxy
from einops import repeat, einsum
from .hyper_block import HyperComputeModule


class Hyper_decoder(nn.Module):
    def __init__(self, d_model=256, d_s=16, expand=1, hyper_threshold=0.8):
        super(Hyper_decoder, self).__init__()
        self.d_state = d_s
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        # -----------------------------------------------------------------------------------
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        self.hyperssm = HyperComputeModule(self.d_inner, self.d_inner, hyper_threshold)
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))  # e * d_model, e * d_model

        self.linear1 = nn.Linear(d_model, 512)
        self.activation = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0)
        self.linear2 = nn.Linear(512, d_model)
        self.dropout2 = nn.Dropout(0)
        self.norm2 = nn.LayerNorm(d_model)

        self.normal_bboxes = nn.Linear(d_model, 4)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, x_0, flow_features, h=None, valid_mask=None):
        """Positions [B, N, L, 4] and trajectory features [B, N, L, D]."""
        b, n_obj, l, p_dim = x_0.shape
        pos = self.gen_sineembed_for_position(
            x_0.reshape(b * n_obj, l, p_dim), dim_per_axis=self.d_model // p_dim
        ).reshape(b, n_obj, l, self.d_model)

        pos_and_motion = self.in_proj(pos)  # shape (b, l, 2 * d_in)
        (pos, pos1) = pos_and_motion.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        pos = F.silu(pos)
        y, h = self.ssm(pos, flow_features, h, valid_mask)
        y = (y * F.silu(pos1))

        y = self.forward_ffn(y)
        y = self.normal_bboxes(y)
        y = F.sigmoid(y)  # if dance sports, need to F.sigmoid()
        if valid_mask is not None:
            y = y.masked_fill(~valid_mask[:, :, None, None], 0)
        return y, h

    def ssm(self, pos_em, flow_em, hidden_state=None, valid_mask=None):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]
        Args:
            pos_em, flow_em: (batch, objects, length, d_model)
    
        Returns:
            output: (batch, objects, length, d_in); state: (batch * objects, d_in, n)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape
        b, n_obj, length, _ = pos_em.shape
        # Cross-object interaction happens before flattening independent temporal scans.
        pos_em = self.hyperssm(pos_em, valid_mask).reshape(b * n_obj, length, d_in)
        flow_em = flow_em.reshape(b * n_obj, length, d_in)

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(
            flow_em)  # (b, l, dt_rank + 2*n) -- flow_em 包含了时序卷积信息的 流特征 --- 从流特征中提取ssm参数信息 -- 用于下一步引导bbox_em的时序变换

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y, hidden_state = self.selective_scan(pos_em, hidden_state, delta, A, B, C,
                                              D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y.reshape(b, n_obj, length, d_in), hidden_state

    def selective_scan(self, u, h, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        u_hyper = u

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplication on B"
        # import pdb;pdb.set_trace()
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u_hyper, 'b l d_in, b l n, b l d_in -> b l d_in n')
        # import pdb;pdb.set_trace()
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        if h is None:
            h = deltaA.new_zeros((b, d_in, n))
        ys = []
        for i in range(l):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = einsum(h, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        # shape (b, l, d_in)
        # y = y + u * D
        y = y + u_hyper

        return y, h

    def gen_sineembed_for_position(self, pos_tensor, dim_per_axis=64):  # normalized cx cy w h
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(dim_per_axis, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / dim_per_axis)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_cx = x_embed[:, :, None] / dim_t
        pos_cy = y_embed[:, :, None] / dim_t
        pos_cx = torch.stack((pos_cx[:, :, 0::2].sin(), pos_cx[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_cy = torch.stack((pos_cy[:, :, 0::2].sin(), pos_cy[:, :, 1::2].cos()), dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_cy, pos_cx), dim=1)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

            pos = torch.cat((pos_cx, pos_cy, pos_w, pos_h), dim=2)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
        return pos


class Time_info_decoder(nn.Module):
    def __init__(self, d_model=256, n_layer=6, d_s=16):
        super(Time_info_decoder, self).__init__()

        hyper_dec = Hyper_decoder(
            d_model=d_model,
            d_s=d_s,
            expand=1
        )
        self.layers = n_layer
        self.head_series = _get_clones(hyper_dec, n_layer)
        self.return_intermediate = True

    def forward(self, x_0, flow_features, h=None, curr_gt=None, active_hyper=False, valid_mask=None):
        if not active_hyper:
            # Each legacy batch item is an independent scene with a single object.
            x_0 = x_0.unsqueeze(1)
            flow_features = flow_features.unsqueeze(1)
            if curr_gt is not None:
                curr_gt = curr_gt.unsqueeze(1)
            if valid_mask is not None:
                valid_mask = valid_mask.reshape(-1, 1)
        b, n_obj, l, p_dim = x_0.shape
        if valid_mask is None:
            valid_mask = torch.ones((b, n_obj), dtype=torch.bool, device=x_0.device)
        valid_mask = valid_mask.to(device=x_0.device, dtype=torch.bool)
        if valid_mask.shape != (b, n_obj):
            raise ValueError('valid_mask must have shape [B, N].')
        if flow_features.shape[:3] != x_0.shape[:3]:
            raise ValueError('Position and trajectory features must share [B, N, L].')
        x_0 = x_0.masked_fill(~valid_mask[:, :, None, None], 0)
        flow_features = flow_features.masked_fill(~valid_mask[:, :, None, None], 0)

        x_inital = x_0
        inter_pred_bboxes = []
        for hyper_head in self.head_series:
            pred_bboxes, h = hyper_head(x_0, flow_features, h, valid_mask)
            if self.training:
                inter_pred_bboxes.append(pred_bboxes)
            x_0 = pred_bboxes.detach()
            h = h.detach()

        if self.training:
            if curr_gt is None or curr_gt.shape != x_inital.shape:
                raise ValueError('Training targets must match position shape [B, N, L, 4].')
            loss = sum(pred.sum() * 0 for pred in inter_pred_bboxes)
            if not valid_mask.any():
                return loss
            # Select real objects before interpolation and either loss computation.
            curr_gt_step = self.interpolation(x_inital[valid_mask], curr_gt[valid_mask])
            for pred, target in zip(inter_pred_bboxes, curr_gt_step):
                pred = pred[valid_mask].reshape(-1, p_dim)
                target = target.reshape(-1, p_dim)
                loss = loss + F.smooth_l1_loss(pred, target)
                loss = loss + generalized_box_iou_loss(
                    box_cxcywh_to_xyxy(pred), box_cxcywh_to_xyxy(target), reduction='mean'
                )
            return loss
        else:
            return pred_bboxes if active_hyper else pred_bboxes.squeeze(1)

    def interpolation(self, x_0, curr_gt_box):
        detla = curr_gt_box - x_0
        detla_dt = detla / self.layers

        curr_gt_step = []
        for i in range(self.layers):
            curr_gt_step.append(x_0 + (i + 1) * detla_dt)

        return curr_gt_step


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
