from __future__ import annotations

import os
import argparse
import datetime
import time
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import default as c
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from post_process import post_process
from utils import load_weights
from rectified_flow_train_posco import MultiScaleRF, msflow_forward, rf_transport, minmax_norm


class PoscoValidationDataset(Dataset):
    """
    Expected structure:
      data/posco/validation/
        00/<one image>
        01/<one image>
        ...

    More generally, it supports multiple images per subfolder too.
    """
    def __init__(self, data_root: str, input_size=(512, 512), img_mean=None, img_std=None):
        self.data_root = data_root
        self.input_size = input_size
        self.img_info_list = self._collect_images(data_root)
        self.transform = T.Compose([
            T.Resize(input_size, InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(img_mean, img_std),
        ])

    @staticmethod
    def _collect_images(data_root: str):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        assert os.path.isdir(data_root), f"Validation folder not found: {data_root}"
        img_info = []
        for subdir in sorted(os.listdir(data_root)):
            subdir_path = os.path.join(data_root, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in sorted(os.listdir(subdir_path)):
                if fname.lower().endswith(exts):
                    img_path = os.path.join(subdir_path, fname)
                    img_info.append((img_path, subdir, fname))
        assert len(img_info) > 0, f"No images found under: {data_root}"
        return img_info

    def __len__(self):
        return len(self.img_info_list)

    def __getitem__(self, idx):
        img_path, subdir, fname = self.img_info_list[idx]
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)
        return x, img_path, subdir, fname


def build_msflow(cfg, ckpt_path: str):
    extractor, output_channels = build_extractor(cfg)
    extractor = extractor.to(cfg.device).eval()

    parallel_flows, fusion_flow = build_msflow_model(cfg, output_channels)
    parallel_flows = [pf.to(cfg.device).eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.to(cfg.device).eval()

    print(f"[INFO] Loading MSFlow checkpoint: {ckpt_path}")
    load_weights(parallel_flows, fusion_flow, ckpt_path)

    return extractor, parallel_flows, fusion_flow


def build_rf_from_batch(device, rf_ckpt_path: str, z_fused_list: List[torch.Tensor], rf_tdims, rf_depths):
    channels_list = [z_fused_list[i].shape[1] for i in [0, 1]]
    rf_model = MultiScaleRF(
        channels_list,
        tdims=rf_tdims,
        depths=rf_depths,
    ).to(device).eval()

    print(f"[INFO] Loading Rectified-Flow checkpoint: {rf_ckpt_path}")
    ckpt = torch.load(rf_ckpt_path, map_location='cpu')
    state = ckpt['rf_model'] if isinstance(ckpt, dict) and 'rf_model' in ckpt else ckpt
    rf_model.load_state_dict(state)
    return rf_model


def anomaly_map_to_bboxes(anomaly_map: np.ndarray, threshold=0.5, min_area=50):
    binary = (anomaly_map >= threshold).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    bboxes = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def draw_bboxes_on_image(img: Image.Image, bboxes, color='red', width=3):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in bboxes:
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return out


def save_outputs(img_tensor: torch.Tensor,
                 anomaly_map: np.ndarray,
                 out_dir: str,
                 fname: str,
                 threshold: float,
                 min_area: int,
                 save_size=(1920, 1080)):
    # Convert normalized tensor -> PIL
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_u8 = ((img_tensor.cpu() * std + mean).clamp(0, 1) * 255).byte()
    img_pil = Image.fromarray(img_u8.permute(1, 2, 0).numpy())

    # Original bboxes are on 512x512 space
    bboxes = anomaly_map_to_bboxes(anomaly_map, threshold=threshold, min_area=min_area)

    # Resize image to final save size: (width, height)
    target_w, target_h = save_size
    src_w, src_h = img_pil.size
    resized_img = img_pil.resize((target_w, target_h), Image.BILINEAR)

    # Scale bbox coordinates to the new size
    scale_x = target_w / src_w
    scale_y = target_h / src_h
    scaled_bboxes = [
        (
            int(x0 * scale_x),
            int(y0 * scale_y),
            int(x1 * scale_x),
            int(y1 * scale_y),
        )
        for x0, y0, x1, y1 in bboxes
    ]

    boxed = draw_bboxes_on_image(resized_img, scaled_bboxes, color='red', width=6)
    boxed.save(os.path.join(out_dir, fname))
    return scaled_bboxes


@torch.no_grad()
def get_final_localization_map(cfg, extractor, parallel_flows, fusion_flow, rf_model, imgs, rf_steps: int):
    # Single forward pass: z_fused_list is identical to what model_forward returns as z_list
    _, z_fused_list, _ = msflow_forward(cfg, extractor, parallel_flows, fusion_flow, imgs, return_pre_fusion=True)

    # Base MSFlow outputs for anomaly_score_map_add (reuse z_fused_list, no second forward)
    size_list = [list(z.shape[-2:]) for z in z_fused_list]
    outputs_list = []
    for z in z_fused_list:
        logp = -0.5 * torch.mean(z ** 2, dim=1)
        outputs_list.append([logp])
    _, anomaly_score_map_add, _ = post_process(cfg, size_list, outputs_list)

    # RF path
    z_rf_in = [z_fused_list[i] for i in [0, 1]]
    z_fused_rect_l01 = rf_transport(rf_model, z_rf_in, steps=rf_steps)

    diff_maps = []
    for lvl in [0, 1]:
        z = z_fused_list[lvl]
        zr = z_fused_rect_l01[lvl]
        logp = -0.5 * torch.mean(z ** 2, dim=1)
        logp_r = -0.5 * torch.mean(zr ** 2, dim=1)
        diff_maps.append((logp_r - logp).detach().cpu())

    d0 = diff_maps[0].unsqueeze(1)
    d1 = diff_maps[1].unsqueeze(1)
    if d0.shape[-2:] != tuple(cfg.input_size):
        d0 = F.interpolate(d0, size=cfg.input_size, mode='bilinear', align_corners=False)
    if d1.shape[-2:] != tuple(cfg.input_size):
        d1 = F.interpolate(d1, size=cfg.input_size, mode='bilinear', align_corners=False)

    rf_map = minmax_norm((d0[:, 0] + d1[:, 0]).numpy())
    return rf_map + anomaly_score_map_add


def main():
    parser = argparse.ArgumentParser(description='Visualize POSCO bounding boxes from current MSFlow+RF localization map')
    parser.add_argument('--data_root', type=str, default='./data/posco/test',
                        help='POSCO validation root containing 00,01,... subfolders')
    parser.add_argument('--output_dir', type=str, default='./results_bboxes_posco_rf_test',
                        help='Where to save images with bounding boxes')
    parser.add_argument('--msflow_ckpt', type=str,
                        default='work_dirs/msflow_wide_resnet50_2_avgpool_pl258/posco/posco/last.pt')
    parser.add_argument('--rf_ckpt', type=str,
                        default='work_dirs/rf_on_msflow_wide_resnet50_2_avgpool_pl258/posco/posco/rf_last.pt')
    parser.add_argument('--threshold', type=float, default=2.5)
    parser.add_argument('--min_area', type=int, default=80,
                        help='Minimum connected region area to keep')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rf_steps', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--rf-tdims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--rf-depths', type=int, nargs='+', default=[3, 3])
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    c.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c.input_size = (512, 512)
    c.img_mean = [0.485, 0.456, 0.406]
    c.img_std = [0.229, 0.224, 0.225]
    c.extractor = 'wide_resnet50_2'
    c.pool_type = 'avg'
    c.parallel_blocks = [2, 5, 8]
    c.c_conds = [64, 64, 64]
    c.clamp_alpha = 1.9
    c.device = c.device

    assert os.path.isfile(args.msflow_ckpt), f"MSFlow checkpoint not found: {args.msflow_ckpt}"
    assert os.path.isfile(args.rf_ckpt), f"RF checkpoint not found: {args.rf_ckpt}"

    dataset = PoscoValidationDataset(args.data_root, input_size=c.input_size,
                                     img_mean=c.img_mean, img_std=c.img_std)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"[INFO] Found {len(dataset)} validation images under {args.data_root}")

    extractor, parallel_flows, fusion_flow = build_msflow(c, args.msflow_ckpt)

    # Initialize rf_model before the timed loop so model loading is excluded from FPS
    print("[INFO] Initializing RF model...")
    init_imgs, _, _, _ = next(iter(loader))
    init_imgs = init_imgs.to(c.device, non_blocking=True)
    with torch.no_grad():
        _, z_fused_list, _ = msflow_forward(c, extractor, parallel_flows, fusion_flow, init_imgs, return_pre_fusion=True)
    rf_model = build_rf_from_batch(c.device, args.rf_ckpt, z_fused_list, args.rf_tdims, args.rf_depths)

    # Pre-create output subdirs to avoid per-image os.makedirs overhead
    seen_dirs = set()

    total_processed = 0
    start = time.time()

    for imgs, img_paths, subdirs, fnames in loader:
        imgs = imgs.to(c.device, non_blocking=True)

        final_maps = get_final_localization_map(c, extractor, parallel_flows, fusion_flow, rf_model, imgs, args.rf_steps)
        total_processed += imgs.shape[0]

        for b in range(imgs.shape[0]):
            fname = fnames[b]
            final_map = final_maps[b]
            if torch.is_tensor(final_map):
                final_map = final_map.detach().cpu().numpy()

            out_dir = os.path.join(args.output_dir, subdirs[b])
            if out_dir not in seen_dirs:
                os.makedirs(out_dir, exist_ok=True)
                seen_dirs.add(out_dir)

            save_outputs(
                img_tensor=imgs[b],
                anomaly_map=final_map,
                out_dir=out_dir,
                fname=fname,
                threshold=args.threshold,
                min_area=args.min_area,
            )

    fps = total_processed / max(time.time() - start, 1e-6)
    print(datetime.datetime.now().strftime('[%Y-%m-%d-%H:%M:%S]'),
          f'Done. Processed {total_processed} images, final model-forward FPS: {fps:.1f}')


if __name__ == '__main__':
    main()
