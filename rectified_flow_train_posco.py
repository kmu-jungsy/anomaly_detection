from __future__ import annotations

import os
import math
import time
import datetime
import argparse
import copy
from typing import List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from datasets import MVTecDataset, VisADataset, POSCODataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from post_process import post_process
from utils import positionalencoding2d, load_weights, t2np
from evaluations import eval_det_loc
from utils import Score_Observer
from noise import pnoise2


def _denorm(x: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    if x.dim() == 4:
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)
    return x * std_t + mean_t


def _renorm(x01: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, device=x01.device, dtype=x01.dtype).view(-1, 1, 1)
    std_t = torch.as_tensor(std, device=x01.device, dtype=x01.dtype).view(-1, 1, 1)
    if x01.dim() == 4:
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)
    return (x01 - mean_t) / std_t


def _rand_uniform(a: float, b: float) -> float:
    return float(torch.empty(1).uniform_(a, b).item())


class DRAEMAnomaly:
    def __init__(self, img_mean, img_std, beta_range=(0.1, 1.0)):
        self.img_mean = img_mean
        self.img_std = img_std
        self.beta_range = beta_range

    def _perlin_mask(self, H, W):
        scale = np.random.uniform(0.05, 0.15)
        thresh = np.random.uniform(0.3, 0.7)
        noise = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                noise[i, j] = pnoise2(i * scale, j * scale)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return (noise > thresh).astype(np.float32)

    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        x01 = _denorm(x_norm, self.img_mean, self.img_std).clamp(0, 1)
        _, H, W = x01.shape
        mask = self._perlin_mask(H, W)
        mask_t = torch.from_numpy(mask).to(x01.device)
        texture = x01.clone()
        aug_ops = [
            lambda x: TF.adjust_brightness(x, np.random.uniform(0.5, 1.5)),
            lambda x: TF.adjust_contrast(x, np.random.uniform(0.5, 1.5)),
            lambda x: TF.adjust_saturation(x, np.random.uniform(0, 2.0)),
            lambda x: TF.adjust_hue(x, np.random.uniform(-0.3, 0.3)),
        ]
        chosen = np.random.choice(len(aug_ops), 3, replace=False)
        for i in chosen:
            texture = aug_ops[i](texture).clamp(0, 1)
        beta = np.random.uniform(*self.beta_range)
        mask3 = mask_t.unsqueeze(0)
        out = (1 - mask3) * x01 + (1 - beta) * mask3 * x01 + beta * mask3 * texture
        out = out.clamp(0, 1)
        return _renorm(out, self.img_mean, self.img_std)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, tdim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.to_scale_shift = nn.Linear(tdim, channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        ss = self.to_scale_shift(t_emb)
        scale, shift = ss.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return x + h


class SmallConvVelocityNet(nn.Module):
    def __init__(self, channels: int, tdim: int = 64, depth: int = 4):
        super().__init__()
        self.time = TimeEmbedding(tdim)
        self.in_proj = nn.Conv2d(channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResBlock(channels, tdim * 4) for _ in range(depth)])
        self.out_proj = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time(t)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_proj(h)


class MultiScaleRF(nn.Module):
    def __init__(self, channels_list: List[int], tdim: int = 64, depth: int = 4,
                 tdims: Optional[List[int]] = None, depths: Optional[List[int]] = None):
        super().__init__()
        if tdims is not None or depths is not None:
            assert tdims is not None and depths is not None
            assert len(tdims) == len(channels_list)
            assert len(depths) == len(channels_list)
            self.nets = nn.ModuleList(
                [SmallConvVelocityNet(c, tdim=int(ti), depth=int(di)) for c, ti, di in zip(channels_list, tdims, depths)]
            )
        else:
            self.nets = nn.ModuleList([SmallConvVelocityNet(c, tdim=tdim, depth=depth) for c in channels_list])

    def forward(self, z_list: List[torch.Tensor], t: torch.Tensor) -> List[torch.Tensor]:
        return [net(z, t) for net, z in zip(self.nets, z_list)]


def msflow_forward(c, extractor, parallel_flows, fusion_flow, image, *, return_pre_fusion: bool = False):
    h_list = extractor(image)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_pre_list = []
    parallel_jac_list = []
    for h, parallel_flow, c_cond in zip(h_list, parallel_flows, c.c_conds):
        y = pool_layer(h)
        B, _, H, W = y.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        z, jac = parallel_flow(y, [cond])
        z_pre_list.append(z)
        parallel_jac_list.append(jac)

    z_fused_list, fuse_jac = fusion_flow(z_pre_list)
    jac = fuse_jac + sum(parallel_jac_list)

    if return_pre_fusion:
        return z_pre_list, z_fused_list, jac
    return z_fused_list, jac


def rf_loss(rf: MultiScaleRF, z0_list: List[torch.Tensor], z1_list: List[torch.Tensor]) -> torch.Tensor:
    B = z0_list[0].shape[0]
    t = torch.rand(B, device=z0_list[0].device)
    loss = 0.0
    for z0, z1, net in zip(z0_list, z1_list, rf.nets):
        t_b = t.view(B, 1, 1, 1)
        zt = (1 - t_b) * z0 + t_b * z1
        v_target = z1 - z0
        v_pred = net(zt, t)
        loss = loss + F.mse_loss(v_pred, v_target)
    return loss / len(z0_list)


@torch.no_grad()
def rf_transport(rf: MultiScaleRF, z0_list: List[torch.Tensor], steps: int = 16) -> List[torch.Tensor]:
    z = [t.clone() for t in z0_list]
    B = z[0].shape[0]
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k / steps, device=z[0].device, dtype=torch.float32)
        v_list = rf(z, t)
        z = [zi + dt * vi for zi, vi in zip(z, v_list)]
    return z


def minmax_norm(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


@torch.no_grad()
def eval_rf_epoch(c, epoch: int, loader: DataLoader, extractor, parallel_flows, fusion_flow,
                  rf_model: MultiScaleRF, ode_steps: int,
                  det_auroc_obs: Score_Observer, loc_auroc_obs: Score_Observer,
                  loc_pro_obs: Score_Observer, pro_eval: bool = False):
    rf_model = rf_model.eval()
    parallel_flows = [pf.eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.eval()

    gt_label_list = []
    gt_mask_list = []
    outputs_list = [list() for _ in parallel_flows]
    size_list = []
    diff_l01_list = [[], []]

    start = time.time()
    for idx, (image, label, mask) in enumerate(loader):
        image = image.to(c.device, non_blocking=True)
        gt_label_list.extend(t2np(label))
        gt_mask_list.extend(t2np(mask))

        z_pre_list, z_fused_list, _ = msflow_forward(c, extractor, parallel_flows, fusion_flow, image, return_pre_fusion=True)
        z_rf_in = [z_fused_list[i] for i in [0, 1]]
        z_fused_rect_l01 = rf_transport(rf_model, z_rf_in, steps=ode_steps)

        z_fused_rect_full = [z.clone() for z in z_fused_list]
        z_fused_rect_full[0] = z_fused_rect_l01[0]
        z_fused_rect_full[1] = z_fused_rect_l01[1]

        for lvl, z in enumerate(z_fused_list):
            if idx == 0:
                size_list.append(list(z.shape[-2:]))
            logp = -0.5 * torch.mean(z ** 2, dim=1)
            outputs_list[lvl].append(logp)

        diff_maps = []
        for lvl in [0, 1]:
            z = z_fused_list[lvl]
            zr = z_fused_rect_full[lvl]
            logp = -0.5 * torch.mean(z ** 2, dim=1)
            logp_r = -0.5 * torch.mean(zr ** 2, dim=1)
            diff_maps.append((logp_r - logp).detach().cpu())

        diff_l01_list[0].append(diff_maps[0])
        diff_l01_list[1].append(diff_maps[1])

    fps = len(loader.dataset) / max(time.time() - start, 1e-6)
    print(datetime.datetime.now().strftime('[%Y-%m-%d-%H:%M:%S]'), f'Epoch {epoch} RF-eval FPS: {fps:.1f}')

    anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

    gt_full = np.asarray(gt_mask_list)
    if gt_full.ndim == 4 and gt_full.shape[1] == 1:
        gt_full = gt_full[:, 0]
    elif gt_full.ndim != 3:
        gt_full = np.squeeze(gt_full)
    H0, W0 = gt_full.shape[-2], gt_full.shape[-1]

    d0 = torch.cat(diff_l01_list[0], dim=0).unsqueeze(1)
    d1 = torch.cat(diff_l01_list[1], dim=0).unsqueeze(1)
    if d0.shape[-2:] != (H0, W0):
        d0 = F.interpolate(d0, size=(H0, W0), mode='bilinear', align_corners=False)
    if d1.shape[-2:] != (H0, W0):
        d1 = F.interpolate(d1, size=(H0, W0), mode='bilinear', align_corners=False)

    anomaly_score_map_add_final = minmax_norm((d0[:, 0] + d1[:, 0]).numpy()) * 6 + anomaly_score_map_add
    rf_map = minmax_norm((d0[:, 0] + d1[:, 0]).numpy())   # [N, H, W]
    rf_score = rf_map.reshape(rf_map.shape[0], -1).max(axis=1)   # [N]
    anomaly_score_final = anomaly_score + rf_score

    return eval_det_loc(
        det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch,
        gt_label_list, anomaly_score_final, gt_mask_list,
        anomaly_score_map_add_final, anomaly_score_map_mul, pro_eval,
    )


def build_args():
    parser = argparse.ArgumentParser(description='Rectified-Flow on top of MSFlow')
    parser.add_argument('--dataset', default='posco', choices=['mvtec', 'visa', 'posco'])
    parser.add_argument('--class-name', default='posco', type=str)
    parser.add_argument("--posco-train-subdir", default="", type=str,
                        help="For POSCO, train RF using only data_path/train/<subdir>.")
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str)
    parser.add_argument('--pool-type', default='avg', type=str)
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, nargs='+')

    parser.add_argument('--msflow-ckpt', default='work_dirs/msflow_wide_resnet50_2_avgpool_pl258/posco/posco/last.pt', type=str)
    parser.add_argument('--train-by-msflow-folder', action='store_true', default=False,
                        help='For POSCO, train one RF model for each MSFlow checkpoint folder under '
                             'msflow-work-dir/msflow-version/dataset/<folder>/msflow-ckpt-name.')
    parser.add_argument('--msflow-class-names', default=None, type=str, nargs='+',
                        help='Optional folder/class names to train. If omitted, all checkpoint folders are used.')
    parser.add_argument('--msflow-work-dir', default='work_dirs', type=str)
    parser.add_argument('--msflow-version', default='msflow_wide_resnet50_2_avgpool_pl258', type=str)
    parser.add_argument('--msflow-ckpt-name', default='last.pt', type=str)

    parser.add_argument('--rf-epochs', default=100, type=int)
    parser.add_argument('--rf-lr', default=5e-4, type=float)
    parser.add_argument('--rf-depth', default=4, type=int)
    parser.add_argument('--rf-tdim', default=64, type=int)
    parser.add_argument('--rf-tdims', default=[128,128], type=int, nargs='+')
    parser.add_argument('--rf-depths', default=[3,3], type=int, nargs='+')
    parser.add_argument('--rf-steps', default=1, type=int)
    parser.add_argument('--rf-ckpt', default='', type=str)

    parser.add_argument('--data-path', default='', type=str)
    parser.add_argument('--work-dir', default='./work_dirs', type=str)
    parser.add_argument('--pro-eval', action='store_true', default=False)
    return parser


def init_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_defaults(c, args):
    c.dataset = args.dataset
    c.class_name = args.class_name
    # POSCO folder-specific training support.
    # datasets.POSCODataset reads c.posco_train_subdir and uses:
    #   data/posco/train/<posco_train_subdir>/*.png
    if c.dataset == 'posco':
        if getattr(args, 'posco_train_subdir', ''):
            c.posco_train_subdir = args.posco_train_subdir
        elif args.class_name != 'posco':
            c.posco_train_subdir = args.class_name
        else:
            c.posco_train_subdir = None
    else:
        c.posco_train_subdir = None
    c.extractor = args.extractor
    c.pool_type = args.pool_type
    c.parallel_blocks = args.parallel_blocks
    c.batch_size = args.batch_size
    c.workers = args.workers
    c.pro_eval = args.pro_eval
    c.pro_eval_interval = 4
    c.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data_path:
        c.data_path = args.data_path
    else:
        if c.dataset == 'mvtec':
            c.data_path = './data/MVTec'
        elif c.dataset == 'visa':
            c.data_path = './data/VisA_pytorch/1cls'
        elif c.dataset == 'posco':
            c.data_path = './data/posco'
        else:
            raise ValueError(f'Unsupported dataset: {c.dataset}')

    c.input_size = (256, 256)
    c.img_mean = [0.485, 0.456, 0.406]
    c.img_std = [0.229, 0.224, 0.225]
    if not hasattr(c, 'c_conds'):
        c.c_conds = [128, 256, 512]

    c.version_name = f"rf_on_msflow_{c.extractor}_{c.pool_type}pool_pl{''.join([str(x) for x in c.parallel_blocks])}"
    c.ckpt_dir = os.path.join(args.work_dir, c.version_name, c.dataset, c.class_name)
    os.makedirs(c.ckpt_dir, exist_ok=True)
    return c


def resolve_msflow_ckpt(args) -> str:
    if args.msflow_ckpt:
        return args.msflow_ckpt
    return os.path.join(
        args.msflow_work_dir,
        args.msflow_version,
        args.dataset,
        args.class_name,
        args.msflow_ckpt_name,
    )


def load_msflow_frozen(c, msflow_ckpt_path: str):
    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
    parallel_flows = [pf.to(c.device).eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.to(c.device).eval()
    _ = load_weights(parallel_flows, fusion_flow, msflow_ckpt_path)

    for p in extractor.parameters():
        p.requires_grad_(False)
    for pf in parallel_flows:
        for p in pf.parameters():
            p.requires_grad_(False)
    for p in fusion_flow.parameters():
        p.requires_grad_(False)
    return extractor, parallel_flows, fusion_flow


def select_dataset_class(name: str):
    if name == 'mvtec':
        return MVTecDataset
    if name == 'visa':
        return VisADataset
    if name == 'posco':
        return POSCODataset
    raise ValueError(f'Unsupported dataset: {name}')


def discover_msflow_class_names(args) -> List[str]:
    """Return folder names that contain an MSFlow checkpoint.

    Expected structure:
        <msflow_work_dir>/<msflow_version>/<dataset>/<class_name>/<msflow_ckpt_name>

    Example:
        work_dirs/msflow_wide_resnet50_2_avgpool_pl258/posco/01/last.pt
        work_dirs/msflow_wide_resnet50_2_avgpool_pl258/posco/02/last.pt
    """
    base_dir = os.path.join(args.msflow_work_dir, args.msflow_version, args.dataset)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f'MSFlow checkpoint folder not found: {base_dir}')

    if args.msflow_class_names:
        class_names = list(args.msflow_class_names)
    else:
        class_names = sorted(
            name for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        )

    valid_class_names = []
    missing = []
    for class_name in class_names:
        ckpt_path = os.path.join(base_dir, class_name, args.msflow_ckpt_name)
        if os.path.isfile(ckpt_path):
            valid_class_names.append(class_name)
        else:
            missing.append(ckpt_path)

    if missing:
        print('[Warning] These MSFlow checkpoints were not found and will be skipped:')
        for path in missing:
            print(f'  - {path}')

    if not valid_class_names:
        raise FileNotFoundError(
            f'No MSFlow checkpoints named {args.msflow_ckpt_name!r} were found under {base_dir}'
        )
    return valid_class_names


def train_rf(args):
    import default as c
    c = resolve_defaults(c, args)
    init_seeds(args.seed)

    Dataset = select_dataset_class(c.dataset)
    train_dataset = Dataset(c, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True)

    msflow_ckpt = resolve_msflow_ckpt(args)
    assert os.path.isfile(msflow_ckpt), f'MSFlow checkpoint not found: {msflow_ckpt}'
    print(f'[MSFlow] load: {msflow_ckpt}')
    extractor, parallel_flows, fusion_flow = load_msflow_frozen(c, msflow_ckpt)

    test_dataset = Dataset(c, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

    det_auroc_obs = Score_Observer('Det.AUROC', args.rf_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', args.rf_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', args.rf_epochs)

    rf_model = None
    optimizer = None
    anomaly_gen = DRAEMAnomaly(c.img_mean, c.img_std, beta_range=(0.1, 1.0))
    
    for epoch in range(args.rf_epochs):
        if rf_model is not None:
            rf_model.train()
        for img, y, _ in train_loader:
            if y.sum().item() != 0:
                img = img[y == 0]
                if img.numel() == 0:
                    continue
            img = img.to(c.device, non_blocking=True)

            p_identity = 0.10
            img_pseudo_list = []
            for x in img:
                if torch.rand(1).item() < p_identity:
                    x_pseudo = x
                else:
                    x_pseudo = anomaly_gen(x)
            
                img_pseudo_list.append(x_pseudo)
            
            img_pseudo = torch.stack(img_pseudo_list, dim=0)

            with torch.no_grad():
                _, z_fused_list, _ = msflow_forward(c, extractor, parallel_flows, fusion_flow, img, return_pre_fusion=True)
                _, z_fused_p_list, _ = msflow_forward(c, extractor, parallel_flows, fusion_flow, img_pseudo, return_pre_fusion=True)

            z_list = [z_fused_list[i] for i in [0, 1]]
            z_p_list = [z_fused_p_list[i] for i in [0, 1]]

            if rf_model is None:
                channels_list = [z.shape[1] for z in z_list]
                if args.rf_tdims is not None or args.rf_depths is not None:
                    rf_model = MultiScaleRF(channels_list, tdims=args.rf_tdims, depths=args.rf_depths).to(c.device)
                else:
                    rf_model = MultiScaleRF(channels_list, tdim=args.rf_tdim, depth=args.rf_depth).to(c.device)
                optimizer = torch.optim.Adam(rf_model.parameters(), lr=args.rf_lr)
                if args.rf_ckpt and os.path.isfile(args.rf_ckpt):
                    ckpt = torch.load(args.rf_ckpt, map_location='cpu')
                    rf_model.load_state_dict(ckpt['rf_model'])
                    optimizer.load_state_dict(ckpt['optimizer'])
                    print(f'[RF] resumed from {args.rf_ckpt}')

            optimizer.zero_grad(set_to_none=True)
            loss = rf_loss(rf_model, z_p_list, z_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rf_model.parameters(), 1.0)
            optimizer.step()

        if rf_model is not None:
            ckpt_last = os.path.join(c.ckpt_dir, 'rf_last.pt')
            torch.save({'rf_model': rf_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, ckpt_last)

    if rf_model is not None:
        print(f'[RF] saved last checkpoint: {os.path.join(c.ckpt_dir, "rf_last.pt")}')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = build_args()
    args = parser.parse_args()

    if args.train_by_msflow_folder:
        class_names = discover_msflow_class_names(args)
        print(f'[RF] train-by-msflow-folder enabled. Found {len(class_names)} MSFlow model(s): {class_names}')
        for i, class_name in enumerate(class_names, start=1):
            run_args = copy.deepcopy(args)
            run_args.class_name = class_name
            run_args.posco_train_subdir = class_name
            # In folder mode, construct the checkpoint path from msflow-work-dir/version/dataset/class-name.
            run_args.msflow_ckpt = ''
            print('\n' + '=' * 80)
            print(f'[RF] ({i}/{len(class_names)}) Train RF for MSFlow class/folder: {class_name}')
            print('=' * 80)
            train_rf(run_args)
    else:
        train_rf(args)


if __name__ == '__main__':
    main()
