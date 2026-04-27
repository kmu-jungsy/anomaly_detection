from __future__ import annotations

import os
import argparse
import datetime
import time
from typing import List, Optional

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


class PoscoTestFolderDataset(Dataset):
    """
    POSCO test dataset for one subfolder/class.

    Expected structure:
      data/posco/test/
        normal/<folder_name>/*.png
        abnormal/<folder_name>/*.png

    Example for folder_name='02':
      data/posco/test/normal/02/*.png
      data/posco/test/abnormal/02/*.png
    """
    def __init__(self, data_root: str, folder_name: str, input_size=(512, 512), img_mean=None, img_std=None):
        self.data_root = data_root
        self.folder_name = folder_name
        self.input_size = input_size
        self.img_info_list = self._collect_images(data_root, folder_name)
        self.transform = T.Compose([
            T.Resize(input_size, InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(img_mean, img_std),
        ])

    @staticmethod
    def _collect_images(data_root: str, folder_name: str):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        assert os.path.isdir(data_root), f"POSCO test folder not found: {data_root}"

        img_info = []
        for label_name in ['normal', 'abnormal']:
            folder_path = os.path.join(data_root, label_name, folder_name)
            if not os.path.isdir(folder_path):
                print(f"[Warning] Missing test folder, skip: {folder_path}")
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(exts):
                    img_path = os.path.join(folder_path, fname)
                    img_info.append((img_path, label_name, folder_name, fname))

        assert len(img_info) > 0, (
            f"No test images found for folder {folder_name!r}. Expected images under:\n"
            f"  {os.path.join(data_root, 'normal', folder_name)}\n"
            f"  {os.path.join(data_root, 'abnormal', folder_name)}"
        )
        return img_info

    def __len__(self):
        return len(self.img_info_list)

    def __getitem__(self, idx):
        img_path, label_name, folder_name, fname = self.img_info_list[idx]
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)
        return x, img_path, label_name, folder_name, fname


class PoscoFlatOrValidationDataset(Dataset):
    """
    Backward-compatible dataset for old single-model mode.
    It scans one-level subfolders under data_root.
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
        assert os.path.isdir(data_root), f"Image folder not found: {data_root}"
        img_info = []
        for subdir in sorted(os.listdir(data_root)):
            subdir_path = os.path.join(data_root, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in sorted(os.listdir(subdir_path)):
                if fname.lower().endswith(exts):
                    img_info.append((os.path.join(subdir_path, fname), subdir, fname))
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
    for i in range(1, num_labels):
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
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_u8 = ((img_tensor.cpu() * std + mean).clamp(0, 1) * 255).byte()
    img_pil = Image.fromarray(img_u8.permute(1, 2, 0).numpy())

    bboxes = anomaly_map_to_bboxes(anomaly_map, threshold=threshold, min_area=min_area)

    target_w, target_h = save_size
    src_w, src_h = img_pil.size
    resized_img = img_pil.resize((target_w, target_h), Image.BILINEAR)

    scale_x = target_w / src_w
    scale_y = target_h / src_h
    scaled_bboxes = [
        (int(x0 * scale_x), int(y0 * scale_y), int(x1 * scale_x), int(y1 * scale_y))
        for x0, y0, x1, y1 in bboxes
    ]

    boxed = draw_bboxes_on_image(resized_img, scaled_bboxes, color='red', width=6)
    boxed.save(os.path.join(out_dir, fname))
    return scaled_bboxes


@torch.no_grad()
def get_final_localization_map(cfg, extractor, parallel_flows, fusion_flow, rf_model, imgs, rf_steps: int):
    _, z_fused_list, _ = msflow_forward(cfg, extractor, parallel_flows, fusion_flow, imgs, return_pre_fusion=True)

    size_list = [list(z.shape[-2:]) for z in z_fused_list]
    outputs_list = []
    for z in z_fused_list:
        logp = -0.5 * torch.mean(z ** 2, dim=1)
        outputs_list.append([logp])
    _, anomaly_score_map_add, _ = post_process(cfg, size_list, outputs_list)

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


def setup_cfg(args, folder_name: Optional[str] = None):
    c.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c.input_size = (512, 512)
    c.img_mean = [0.485, 0.456, 0.406]
    c.img_std = [0.229, 0.224, 0.225]
    c.extractor = args.extractor
    c.pool_type = args.pool_type
    c.parallel_blocks = args.parallel_blocks
    c.c_conds = args.c_conds
    c.clamp_alpha = args.clamp_alpha
    c.dataset = 'posco'
    if folder_name is not None:
        c.class_name = folder_name
        c.posco_train_subdir = folder_name
    return c


def discover_folder_names(args) -> List[str]:
    if args.folder_names:
        return list(args.folder_names)

    msflow_base = os.path.join(args.msflow_work_dir, args.msflow_version, 'posco')
    rf_base = os.path.join(args.rf_work_dir, args.rf_version, 'posco')
    normal_base = os.path.join(args.data_root, 'normal')
    abnormal_base = os.path.join(args.data_root, 'abnormal')

    candidate_sets = []
    for base in [msflow_base, rf_base, normal_base, abnormal_base]:
        if os.path.isdir(base):
            candidate_sets.append({d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))})
        else:
            print(f"[Warning] Folder not found while discovering subfolders: {base}")

    if not candidate_sets:
        raise FileNotFoundError('Could not discover any folder names. Use --folder-names 01 02 ...')

    folder_names = sorted(set.intersection(*candidate_sets)) if len(candidate_sets) > 1 else sorted(candidate_sets[0])

    valid = []
    skipped = []
    for folder in folder_names:
        msflow_ckpt = os.path.join(msflow_base, folder, args.msflow_ckpt_name)
        rf_ckpt = os.path.join(rf_base, folder, args.rf_ckpt_name)
        has_normal = os.path.isdir(os.path.join(normal_base, folder))
        has_abnormal = os.path.isdir(os.path.join(abnormal_base, folder))
        if os.path.isfile(msflow_ckpt) and os.path.isfile(rf_ckpt) and (has_normal or has_abnormal):
            valid.append(folder)
        else:
            skipped.append((folder, msflow_ckpt, rf_ckpt, has_normal, has_abnormal))

    if skipped:
        print('[Warning] Some folders were skipped because checkpoint or test folder was missing:')
        for folder, ms_ckpt, rf_ckpt, has_normal, has_abnormal in skipped:
            print(f"  - {folder}: msflow={os.path.isfile(ms_ckpt)}, rf={os.path.isfile(rf_ckpt)}, "
                  f"normal={has_normal}, abnormal={has_abnormal}")

    if not valid:
        raise FileNotFoundError('No valid folders found with both checkpoints and test images.')
    return valid


def run_one_folder(args, folder_name: str):
    cfg = setup_cfg(args, folder_name)

    msflow_ckpt = os.path.join(
        args.msflow_work_dir, args.msflow_version, 'posco', folder_name, args.msflow_ckpt_name
    )
    rf_ckpt = os.path.join(
        args.rf_work_dir, args.rf_version, 'posco', folder_name, args.rf_ckpt_name
    )

    assert os.path.isfile(msflow_ckpt), f"MSFlow checkpoint not found: {msflow_ckpt}"
    assert os.path.isfile(rf_ckpt), f"RF checkpoint not found: {rf_ckpt}"

    dataset = PoscoTestFolderDataset(
        args.data_root,
        folder_name=folder_name,
        input_size=cfg.input_size,
        img_mean=cfg.img_mean,
        img_std=cfg.img_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print('\n' + '=' * 80)
    print(f"[INFO] Visualize folder: {folder_name}")
    print(f"[INFO] Test images: {len(dataset)}")
    print(f"[INFO] MSFlow: {msflow_ckpt}")
    print(f"[INFO] RF:     {rf_ckpt}")
    print('=' * 80)

    extractor, parallel_flows, fusion_flow = build_msflow(cfg, msflow_ckpt)

    print('[INFO] Initializing RF model...')
    init_imgs, *_ = next(iter(loader))
    init_imgs = init_imgs.to(cfg.device, non_blocking=True)
    with torch.no_grad():
        _, z_fused_list, _ = msflow_forward(cfg, extractor, parallel_flows, fusion_flow, init_imgs, return_pre_fusion=True)
    rf_model = build_rf_from_batch(cfg.device, rf_ckpt, z_fused_list, args.rf_tdims, args.rf_depths)

    seen_dirs = set()
    total_processed = 0
    start = time.time()

    for imgs, img_paths, label_names, folder_names, fnames in loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        final_maps = get_final_localization_map(cfg, extractor, parallel_flows, fusion_flow, rf_model, imgs, args.rf_steps)
        total_processed += imgs.shape[0]

        for b in range(imgs.shape[0]):
            fname = fnames[b]
            final_map = final_maps[b]
            if torch.is_tensor(final_map):
                final_map = final_map.detach().cpu().numpy()

            # Save separately to avoid name collision between normal/abnormal.
            out_dir = os.path.join(args.output_dir, folder_name, label_names[b])
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
          f'Folder {folder_name}: processed {total_processed} images, FPS: {fps:.1f}')

    del extractor, parallel_flows, fusion_flow, rf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_model(args):
    cfg = setup_cfg(args, None)
    assert os.path.isfile(args.msflow_ckpt), f"MSFlow checkpoint not found: {args.msflow_ckpt}"
    assert os.path.isfile(args.rf_ckpt), f"RF checkpoint not found: {args.rf_ckpt}"

    dataset = PoscoFlatOrValidationDataset(args.data_root, input_size=cfg.input_size,
                                           img_mean=cfg.img_mean, img_std=cfg.img_std)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    print(f"[INFO] Found {len(dataset)} images under {args.data_root}")
    extractor, parallel_flows, fusion_flow = build_msflow(cfg, args.msflow_ckpt)

    print('[INFO] Initializing RF model...')
    init_imgs, _, _, _ = next(iter(loader))
    init_imgs = init_imgs.to(cfg.device, non_blocking=True)
    with torch.no_grad():
        _, z_fused_list, _ = msflow_forward(cfg, extractor, parallel_flows, fusion_flow, init_imgs, return_pre_fusion=True)
    rf_model = build_rf_from_batch(cfg.device, args.rf_ckpt, z_fused_list, args.rf_tdims, args.rf_depths)

    seen_dirs = set()
    total_processed = 0
    start = time.time()

    for imgs, img_paths, subdirs, fnames in loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        final_maps = get_final_localization_map(cfg, extractor, parallel_flows, fusion_flow, rf_model, imgs, args.rf_steps)
        total_processed += imgs.shape[0]

        for b in range(imgs.shape[0]):
            final_map = final_maps[b]
            if torch.is_tensor(final_map):
                final_map = final_map.detach().cpu().numpy()

            out_dir = os.path.join(args.output_dir, subdirs[b])
            if out_dir not in seen_dirs:
                os.makedirs(out_dir, exist_ok=True)
                seen_dirs.add(out_dir)

            save_outputs(imgs[b], final_map, out_dir, fnames[b], args.threshold, args.min_area)

    fps = total_processed / max(time.time() - start, 1e-6)
    print(datetime.datetime.now().strftime('[%Y-%m-%d-%H:%M:%S]'),
          f'Done. Processed {total_processed} images, FPS: {fps:.1f}')


def main():
    parser = argparse.ArgumentParser(description='Visualize POSCO bounding boxes from MSFlow+RF localization map')
    parser.add_argument('--data_root', type=str, default='./data/posco/test',
                        help='POSCO test root containing normal/<folder> and abnormal/<folder>')
    parser.add_argument('--output_dir', type=str, default='./results_bboxes_posco_rf_test',
                        help='Where to save images with bounding boxes')

    # Old single-model mode arguments.
    parser.add_argument('--msflow_ckpt', type=str,
                        default='work_dirs/msflow_wide_resnet50_2_avgpool_pl258/posco/posco/last.pt')
    parser.add_argument('--rf_ckpt', type=str,
                        default='work_dirs/rf_on_msflow_wide_resnet50_2_avgpool_pl258/posco/posco/rf_last.pt')

    # New folder-by-folder mode arguments.
    parser.add_argument('--visualize-by-folder', action='store_true', default=False,
                        help='Run each POSCO subfolder with its matching MSFlow and RF checkpoints.')
    parser.add_argument('--folder-names', type=str, nargs='+', default=None,
                        help='Optional folder names to run, e.g., --folder-names 01 02 05. If omitted, auto-discover.')
    parser.add_argument('--msflow-work-dir', type=str, default='work_dirs')
    parser.add_argument('--msflow-version', type=str, default='msflow_wide_resnet50_2_avgpool_pl258')
    parser.add_argument('--msflow-ckpt-name', type=str, default='last.pt')
    parser.add_argument('--rf-work-dir', type=str, default='work_dirs')
    parser.add_argument('--rf-version', type=str, default='rf_on_msflow_wide_resnet50_2_avgpool_pl258')
    parser.add_argument('--rf-ckpt-name', type=str, default='rf_last.pt')

    parser.add_argument('--threshold', type=float, default=2.5)
    parser.add_argument('--min_area', type=int, default=80,
                        help='Minimum connected region area to keep')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rf_steps', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--rf-tdims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--rf-depths', type=int, nargs='+', default=[3, 3])
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str)
    parser.add_argument('--pool-type', default='avg', type=str)
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, nargs='+')
    parser.add_argument('--c-conds', default=[64, 64, 64], type=int, nargs='+')
    parser.add_argument('--clamp-alpha', default=1.9, type=float)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize_by_folder:
        folder_names = discover_folder_names(args)
        print(f"[INFO] visualize-by-folder enabled. Found {len(folder_names)} folder(s): {folder_names}")
        for folder_name in folder_names:
            run_one_folder(args, folder_name)
    else:
        run_single_model(args)


if __name__ == '__main__':
    main()
