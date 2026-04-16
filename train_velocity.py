import os
import random
import argparse
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler

from datasets import MVTecDataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from models.velocity_unet import VelocityUNet
from utils import load_weights, Score_Observer, t2np
from evaluations import eval_det_loc
from post_process import post_process as msflow_post_process  # MSFlow official post-process
from train import model_forward  # MSFlow forward (extractor + parallel + fusion)


def init_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _denorm_to_01(x_norm: torch.Tensor, mean, std):
    """(B,C,H,W) normalized -> [0,1]"""
    mean_t = torch.as_tensor(mean, device=x_norm.device, dtype=x_norm.dtype).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, device=x_norm.device, dtype=x_norm.dtype).view(1, -1, 1, 1)
    x = x_norm * std_t + mean_t
    return x.clamp_(0.0, 1.0)


def _norm_from_01(x_01: torch.Tensor, mean, std):
    """(B,C,H,W) [0,1] -> normalized"""
    mean_t = torch.as_tensor(mean, device=x_01.device, dtype=x_01.dtype).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, device=x_01.device, dtype=x_01.dtype).view(1, -1, 1, 1)
    return (x_01 - mean_t) / std_t


@torch.no_grad()
def cutpaste_batch(
    x_norm: torch.Tensor,
    mean,
    std,
    *,
    area_ratio_min: float = 0.02,
    area_ratio_max: float = 0.15,
    aspect_min: float = 0.3,
    aspect_max: float = 3.3,
    rotate: bool = True,
    jitter: float = 0.0,
    prob: float = 0.5,
):
    """CutPaste (basic) augmentation.

    - Cut a rectangular patch from the image, optionally apply light appearance jitter,
      optionally rotate, and paste it at another random location.
    - Operates in [0,1] space internally, then returns normalized tensor.
    """
    assert x_norm.dim() == 4 and x_norm.size(1) == 3, "Expected (B,3,H,W)"
    x = _denorm_to_01(x_norm, mean, std)
    B, C, H, W = x.shape
    out = x.clone()

    # Precompute area bounds in pixels
    area_min = area_ratio_min * H * W
    area_max = area_ratio_max * H * W

    for b in range(B):
        if prob < 1.0 and torch.rand((), device=x.device) > prob:
            continue

        # Sample patch area + aspect ratio
        area = float(torch.empty((), device=x.device).uniform_(area_min, area_max))
        aspect = float(torch.exp(torch.empty((), device=x.device).uniform_(math.log(aspect_min), math.log(aspect_max))))

        ph = int(round(math.sqrt(area / max(aspect, 1e-6))))
        pw = int(round(ph * aspect))
        ph = max(1, min(ph, H - 1))
        pw = max(1, min(pw, W - 1))

        y0 = int(torch.randint(0, H - ph + 1, (1,), device=x.device))
        x0 = int(torch.randint(0, W - pw + 1, (1,), device=x.device))
        patch = out[b : b + 1, :, y0 : y0 + ph, x0 : x0 + pw].clone()

        if jitter and jitter > 0:
            bright = float(torch.empty((), device=x.device).uniform_(1.0 - jitter, 1.0 + jitter))
            contrast = float(torch.empty((), device=x.device).uniform_(1.0 - jitter, 1.0 + jitter))
            patch = (patch * contrast + (bright - 1.0)).clamp_(0.0, 1.0)

        if rotate:
            k = int(torch.randint(0, 4, (1,), device=x.device))
            patch = torch.rot90(patch, k=k, dims=(-2, -1))

        ph2, pw2 = patch.shape[-2:]
        y1 = int(torch.randint(0, H - ph2 + 1, (1,), device=x.device))
        x1 = int(torch.randint(0, W - pw2 + 1, (1,), device=x.device))
        out[b : b + 1, :, y1 : y1 + ph2, x1 : x1 + pw2] = patch

    return _norm_from_01(out, mean, std)


def rectify_latent(z0: torch.Tensor, vel_net: nn.Module, steps: int = 32) -> torch.Tensor:
    """Forward Euler solve: dZ/dt = v_theta(Z,t), t:0->1."""
    z = z0
    dt = 1.0 / float(steps)
    B = z.shape[0]
    for i in range(steps):
        t = torch.full((B,), float(i) / float(steps), device=z.device, dtype=z.dtype)
        v = vel_net(z, t)
        z = z + dt * v
    return z


def build_and_load_msflow(c, ckpt_path: str):
    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
    parallel_flows = [p.to(c.device).eval() for p in parallel_flows]
    fusion_flow = fusion_flow.to(c.device).eval()

    load_weights(parallel_flows, fusion_flow, ckpt_path)

    # freeze
    for p in extractor.parameters():
        p.requires_grad_(False)
    for pf in parallel_flows:
        for p in pf.parameters():
            p.requires_grad_(False)
    for p in fusion_flow.parameters():
        p.requires_grad_(False)

    return extractor, parallel_flows, fusion_flow


def parse_args():
    parser = argparse.ArgumentParser(
        "Train rectified-flow velocity nets on MSFlow fused latents (CutPaste pseudo anomalies, per-scale RF)"
    )

    # Data / MSFlow ckpt
    parser.add_argument("--data_path", type=str, default="./data/MVTec",
                        help="Path to MVTec root. Should contain class folders.")
    parser.add_argument("--msflow_ckpt_root", type=str,
                        default="work_dirs/msflow_wide_resnet50_2_avgpool_pl258/mvtec",
                        help="Directory that contains per-class folders with best_loc_auroc.pt")
    parser.add_argument("--class_name", type=str, required=True,
                        help="MVTec class name (e.g., bottle, cable, hazelnut, ...)")
    parser.add_argument("--msflow_ckpt_name", type=str, default="best_loc_auroc.pt")

    # Velocity net + RF
    parser.add_argument("--velocity_out_root", type=str, default=None,
                        help="Where to save velocity checkpoints. Default: msflow_ckpt_root/<class_name>/")
    parser.add_argument("--steps", type=int, default=32,
                        help="Euler steps for rectification (train/test uses same)")
    parser.add_argument("--diff_weight", type=float, default=0.1,
                        help="final_add = base_add + diff_weight * diff_add")

    # CutPaste params (ONLY pseudo anomaly source)
    parser.add_argument("--cp_area_min", type=float, default=0.02,
                        help="Min patch area ratio (fraction of H*W)")
    parser.add_argument("--cp_area_max", type=float, default=0.15,
                        help="Max patch area ratio (fraction of H*W)")
    parser.add_argument("--cp_aspect_min", type=float, default=0.3,
                        help="Min aspect ratio for patch")
    parser.add_argument("--cp_aspect_max", type=float, default=3.3,
                        help="Max aspect ratio for patch")
    parser.add_argument("--cp_rotate", action=argparse.BooleanOptionalAction, default=False,
                        help="Random 0/90/180/270 rotation on patch before pasting")
    parser.add_argument("--cp_jitter", type=float, default=0.0,
                        help="Brightness/contrast jitter strength for patch (0 disables)")
    parser.add_argument("--cp_prob", type=float, default=0.5,
                        help="Probability to apply CutPaste per image")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=False)

    # U-Net size
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Eval frequency during training
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Run evaluation every N epochs during training (default: 1). Set 0 to disable.")

    # Eval (optional)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--velocity_ckpt", type=str, default="velocity_best.pt",
                        help="Filename (or full path) to load for eval_only. If relative, it's resolved in velocity_out_dir.")
    parser.add_argument("--seed", type=int, default=9826)
    parser.add_argument("--device", type=str, default="cuda")

    # MSFlow config (must match checkpoint)
    parser.add_argument("--extractor", type=str, default="wide_resnet50_2")
    parser.add_argument("--pool_type", type=str, default="avg", choices=["avg", "max", "none"])
    parser.add_argument("--parallel_blocks", type=int, nargs="+", default=[2, 5, 8])
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--input_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--top_k", type=float, default=0.03)

    return parser.parse_args()


@torch.no_grad()
def run_eval(
    c,
    test_loader,
    extractor,
    parallel_flows,
    fusion_flow,
    vel_nets: nn.ModuleList,   # [vel_net1, vel_net2, vel_net3]
    steps: int,
    diff_weight: float = 0.1,
    amp: bool = False,
):
    """Eval: per-scale RF on *fused* latents (NO upsample+concat)."""
    vel_nets.eval()

    gt_label_list = []
    gt_mask_list = []

    size_list = None
    outputs_raw_list = [list() for _ in parallel_flows]
    outputs_corr_list = [list() for _ in parallel_flows]

    start = time.time()
    for _, (image, label, mask) in enumerate(test_loader):
        image = image.to(c.device, non_blocking=True)
        gt_label_list.extend(t2np(label))
        gt_mask_list.extend(t2np(mask))

        with torch.amp.autocast('cuda', enabled=amp):
            # 1) raw fused latents
            z_list_raw, _ = model_forward(c, extractor, parallel_flows, fusion_flow, image)

            # 2) rectify per scale (NO upsample/concat)
            z_list_corr = [
                rectify_latent(z_raw, vel_nets[lvl], steps=steps)
                for lvl, z_raw in enumerate(z_list_raw)
            ]

            if size_list is None:
                size_list = [list(z.shape[-2:]) for z in z_list_raw]

            for lvl, (z_raw, z_corr) in enumerate(zip(z_list_raw, z_list_corr)):
                logp_raw  = -0.5 * torch.mean((z_raw  ** 2).float(), dim=1)
                logp_corr = -0.5 * torch.mean((z_corr ** 2).float(), dim=1)
                outputs_raw_list[lvl].append(logp_raw.detach())
                outputs_corr_list[lvl].append(logp_corr.detach())

    fps = len(test_loader.dataset) / max((time.time() - start), 1e-9)
    print('fps : {:.1f}'.format(fps))
    assert size_list is not None, "Empty test loader?"

    base_score, base_add, base_mul = msflow_post_process(c, size_list, outputs_raw_list)

    outputs_diff_list = []
    for lvl in range(len(outputs_raw_list)):
        lvl_list = []
        for raw_map, corr_map in zip(outputs_raw_list[lvl], outputs_corr_list[lvl]):
            lvl_list.append(raw_map - corr_map)  # logp_raw - logp_corr
        outputs_diff_list.append(lvl_list)

    diff_score, diff_add, diff_mul = msflow_post_process(c, size_list, outputs_diff_list)

    anomaly_score = base_score + diff_score
    anomaly_map_add = base_add + diff_weight * diff_add
    anomaly_map_mul = base_mul + diff_mul

    det_obs = Score_Observer("Det.AUROC", 1)
    loc_obs = Score_Observer("Loc.AUROC", 1)
    pro_obs = Score_Observer("Loc.PRO", 1)

    det, loc, pro, *_ = eval_det_loc(
        det_obs, loc_obs, pro_obs, 1,
        gt_label_list, anomaly_score,
        gt_mask_list, anomaly_map_add, anomaly_map_mul,
        pro_eval=False,
    )
    print(f"[Eval][{c.class_name}] Det.AUROC={det:.3f}  Loc.AUROC={loc:.3f}")


def main():
    args = parse_args()
    init_seeds(args.seed)

    import default as c
    c.dataset = "mvtec"
    c.data_path = args.data_path
    c.class_name = args.class_name
    c.extractor = args.extractor
    c.pool_type = args.pool_type
    c.parallel_blocks = args.parallel_blocks
    c.clamp_alpha = args.clamp_alpha
    c.input_size = tuple(args.input_size)
    c.top_k = args.top_k
    c.workers = args.workers
    c.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    msflow_ckpt = os.path.join(args.msflow_ckpt_root, args.class_name, args.msflow_ckpt_name)
    if not os.path.exists(msflow_ckpt):
        raise FileNotFoundError(f"MSFlow checkpoint not found: {msflow_ckpt}")

    extractor, parallel_flows, fusion_flow = build_and_load_msflow(c, msflow_ckpt)

    train_dataset = MVTecDataset(c, is_train=True)
    test_dataset = MVTecDataset(c, is_train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Build per-scale velocity nets based on *fused* latent channels
    with torch.no_grad():
        x0, _, _ = next(iter(train_loader))
        x0 = x0.to(c.device)
        z_list_fused, _ = model_forward(c, extractor, parallel_flows, fusion_flow, x0)
        in_ch_list = [int(z.shape[1]) for z in z_list_fused]   # z1,z2,z3 channel sizes

    vel_nets = nn.ModuleList([
        VelocityUNet(in_channels=ch, base_channels=args.base_channels, dropout=args.dropout).to(c.device)
        for ch in in_ch_list
    ])

    velocity_out_dir = (
        os.path.join(args.msflow_ckpt_root, args.class_name)
        if args.velocity_out_root is None
        else os.path.join(args.velocity_out_root, args.class_name)
    )
    os.makedirs(velocity_out_dir, exist_ok=True)
    best_path = os.path.join(velocity_out_dir, "velocity_best.pt")
    last_path = os.path.join(velocity_out_dir, "velocity_last.pt")

    # Eval-only
    if args.eval_only:
        ckpt_path = args.velocity_ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(velocity_out_dir, ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu")

        if "velocity_nets" in state:
            sd_list = state["velocity_nets"]
            assert len(sd_list) == len(vel_nets), f"ckpt has {len(sd_list)} nets, code expects {len(vel_nets)}"
            for vnet, sd in zip(vel_nets, sd_list):
                vnet.load_state_dict(sd, strict=True)
        else:
            # legacy fallback (single net checkpoint)
            vel_nets[0].load_state_dict(state["velocity_net"], strict=True)
            print("[WARN] Loaded legacy single-net checkpoint into vel_nets[0] only.")

        vel_nets.eval()
        run_eval(
            c, test_loader, extractor, parallel_flows, fusion_flow, vel_nets,
            args.steps, diff_weight=args.diff_weight, amp=args.amp
        )
        return

    optimizer = torch.optim.AdamW(vel_nets.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        vel_nets.train()
        running = 0.0
        n = 0

        for image, _, _ in train_loader:
            image = image.to(c.device, non_blocking=True)

            # 1) normal (clean) fused latents
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=args.amp):
                z_list_clean, _ = model_forward(c, extractor, parallel_flows, fusion_flow, image)

            # 2) CutPaste pseudo-anomaly images -> fused anomaly latents
            x_cp = cutpaste_batch(
                image,
                c.img_mean,
                c.img_std,
                area_ratio_min=args.cp_area_min,
                area_ratio_max=args.cp_area_max,
                aspect_min=args.cp_aspect_min,
                aspect_max=args.cp_aspect_max,
                rotate=args.cp_rotate,
                jitter=args.cp_jitter,
                prob=args.cp_prob,
            )
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=args.amp):
                z_list_0, _ = model_forward(c, extractor, parallel_flows, fusion_flow, x_cp)

            # 3) rectified-flow training pairs per scale: z0 (anom) -> z1 (normal)
            B = z_list_clean[0].shape[0]
            t = torch.rand((B,), device=c.device, dtype=z_list_clean[0].dtype)  # shared across scales

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                losses = []
                for lvl, (z_clean, z0) in enumerate(zip(z_list_clean, z_list_0)):
                    z_t = t[:, None, None, None] * z_clean + (1.0 - t[:, None, None, None]) * z0
                    target = z_clean - z0
                    pred = vel_nets[lvl](z_t, t)
                    losses.append(F.mse_loss(pred, target))
                loss = sum(losses) / max(len(losses), 1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * B
            n += B

        mean_loss = running / max(n, 1)
        print(f"[Velocity][{args.class_name}] Epoch {epoch}/{args.epochs}  loss={mean_loss:.6f}")

        # save last
        torch.save(
            {
                "epoch": epoch,
                "velocity_nets": [vn.state_dict() for vn in vel_nets],
                "in_channels": in_ch_list,
                "args": vars(args),
            },
            last_path
        )

        # save best
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {
                    "epoch": epoch,
                    "velocity_nets": [vn.state_dict() for vn in vel_nets],
                    "in_channels": in_ch_list,
                    "args": vars(args),
                },
                best_path
            )
            print(f"  -> saved best to {best_path}")

        # periodic eval
        if args.eval_every > 0 and (epoch % args.eval_every == 0):
            vel_nets.eval()
            run_eval(
                c, test_loader, extractor, parallel_flows, fusion_flow, vel_nets,
                args.steps, diff_weight=args.diff_weight, amp=args.amp
            )


if __name__ == "__main__":
    main()
