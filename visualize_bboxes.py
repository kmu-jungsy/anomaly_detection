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



def load_keep_mask(mask_dir: str, folder_name: str, input_size, threshold: int = 10) -> torch.Tensor:
    """Load folder-specific ROI mask as a tensor.

    White/non-black area -> 1.0, keep original image pixels.
    Black area           -> 0.0, make image pixels black.

    Returned shape: [3, H, W]
    """
    mask_path = os.path.join(mask_dir, f"{folder_name}_mask.jpg")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask file not found for folder {folder_name}: {mask_path}")

    mask_img = Image.open(mask_path).convert('L')
    mask_img = mask_img.resize((input_size[1], input_size[0]), Image.BILINEAR)
    mask_np = np.array(mask_img, dtype=np.uint8)

    keep_np = (mask_np > threshold).astype(np.float32)
    keep_tensor = torch.from_numpy(keep_np).unsqueeze(0).repeat(3, 1, 1)
    return keep_tensor


def apply_keep_mask_to_pil(img: Image.Image, mask_dir: str, folder_name: str, input_size, threshold: int = 10) -> Image.Image:
    """Return a masked copy of img for model input.

    The original PIL image passed to this function is not modified.
    """
    img_resized = img.copy().resize((input_size[1], input_size[0]), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.uint8)

    mask_path = os.path.join(mask_dir, f"{folder_name}_mask.jpg")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask file not found for folder {folder_name}: {mask_path}")

    mask_img = Image.open(mask_path).convert('L')
    mask_img = mask_img.resize((input_size[1], input_size[0]), Image.BILINEAR)
    mask_np = np.array(mask_img, dtype=np.uint8)

    # mask white area -> keep original pixels
    # mask black area -> set pixels to black
    keep = mask_np > threshold
    masked_np = img_np.copy()
    masked_np[~keep] = 0
    return Image.fromarray(masked_np)


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
    def __init__(self, data_root: str, folder_name: str, input_size=(512, 512), img_mean=None, img_std=None,
                 apply_test_mask: bool = False, mask_dir: str = './mask', mask_threshold: int = 10,
                 save_masked_debug: bool = False, masked_debug_dir: str = './debug_posco_test_mask'):
        self.data_root = data_root
        self.folder_name = folder_name
        self.input_size = input_size
        self.apply_test_mask = apply_test_mask
        self.mask_dir = mask_dir
        self.mask_threshold = mask_threshold
        self.save_masked_debug = save_masked_debug
        self.masked_debug_dir = masked_debug_dir
        self._saved_debug = False
        self.img_info_list = self._collect_images(data_root, folder_name)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(img_mean, img_std)

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
        original_img = Image.open(img_path).convert('RGB')

        # Use a masked copy only for model input. The original image file is never modified.
        if self.apply_test_mask:
            model_img = apply_keep_mask_to_pil(
                original_img,
                mask_dir=self.mask_dir,
                folder_name=folder_name,
                input_size=self.input_size,
                threshold=self.mask_threshold,
            )

            if self.save_masked_debug and not self._saved_debug:
                debug_dir = os.path.join(self.masked_debug_dir, folder_name, label_name)
                os.makedirs(debug_dir, exist_ok=True)
                stem, ext = os.path.splitext(fname)
                ext = ext if ext else '.jpg'
                model_img.save(os.path.join(debug_dir, f"{stem}_masked_input{ext}"))
                self._saved_debug = True
        else:
            model_img = original_img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)

        x = self.normalize(self.to_tensor(model_img))
        return x, img_path, label_name, folder_name, fname


class PoscoFlatTestDataset(Dataset):
    """
    POSCO flat test dataset for the whole POSCO model.

    Expected structure:
      data/posco/test/
        normal/*.jpg|jpeg|png|bmp|tif|tiff|webp
        abnormal/*.jpg|jpeg|png|bmp|tif|tiff|webp

    This also supports nested folders under normal/abnormal, but outputs are grouped
    only by label name: normal or abnormal.
    """
    def __init__(self, data_root: str, input_size=(512, 512), img_mean=None, img_std=None,
                 apply_test_mask: bool = False, mask_dir: str = './mask', mask_threshold: int = 10,
                 save_masked_debug: bool = False, masked_debug_dir: str = './debug_posco_test_mask'):
        self.data_root = data_root
        self.input_size = input_size
        self.apply_test_mask = apply_test_mask
        self.mask_dir = mask_dir
        self.mask_threshold = mask_threshold
        self.save_masked_debug = save_masked_debug
        self.masked_debug_dir = masked_debug_dir
        self._saved_debug = False
        self.img_info_list = self._collect_images(data_root)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(img_mean, img_std)

    @staticmethod
    def _collect_images(data_root: str):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        assert os.path.isdir(data_root), f"POSCO test root not found: {data_root}"

        img_info = []
        for label_name in ['normal', 'abnormal']:
            label_dir = os.path.join(data_root, label_name)
            assert os.path.isdir(label_dir), f"Missing POSCO test folder: {label_dir}"

            for dirpath, _, filenames in os.walk(label_dir):
                rel_dir = os.path.relpath(dirpath, label_dir)
                rel_prefix = '' if rel_dir == '.' else rel_dir.replace(os.sep, '_') + '_'
                for fname in sorted(filenames):
                    if not fname.lower().endswith(exts):
                        continue
                    img_path = os.path.join(dirpath, fname)
                    # Keep flat output safe even when nested input folders have same filenames.
                    save_fname = rel_prefix + fname
                    folder_name = rel_dir.split(os.sep)[0] if rel_dir != '.' else None
                    img_info.append((img_path, label_name, folder_name, save_fname))

        assert len(img_info) > 0, (
            f"No POSCO test images found. Expected images under:\n"
            f"  {os.path.join(data_root, 'normal')}\n"
            f"  {os.path.join(data_root, 'abnormal')}"
        )
        normal_count = sum(1 for _, label, _, _ in img_info if label == 'normal')
        abnormal_count = sum(1 for _, label, _, _ in img_info if label == 'abnormal')
        print(f"[INFO] POSCO flat test images: normal={normal_count}, abnormal={abnormal_count}")
        return img_info

    def __len__(self):
        return len(self.img_info_list)

    def __getitem__(self, idx):
        img_path, label_name, folder_name, save_fname = self.img_info_list[idx]
        original_img = Image.open(img_path).convert('RGB')

        if self.apply_test_mask:
            if folder_name is None:
                raise ValueError(
                    f"Cannot choose mask for flat test image without subfolder: {img_path}. "
                    "Use data/posco/test/{normal,abnormal}/02/*.jpg style folders, "
                    "or use --visualize-by-folder."
                )
            model_img = apply_keep_mask_to_pil(
                original_img,
                mask_dir=self.mask_dir,
                folder_name=folder_name,
                input_size=self.input_size,
                threshold=self.mask_threshold,
            )

            if self.save_masked_debug and not self._saved_debug:
                debug_dir = os.path.join(self.masked_debug_dir, folder_name, label_name)
                os.makedirs(debug_dir, exist_ok=True)
                stem, ext = os.path.splitext(save_fname)
                ext = ext if ext else '.jpg'
                model_img.save(os.path.join(debug_dir, f"{stem}_masked_input{ext}"))
                self._saved_debug = True
        else:
            model_img = original_img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)

        x = self.normalize(self.to_tensor(model_img))
        return x, img_path, label_name, folder_name, save_fname

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


def save_heatmap_outputs(anomaly_map: np.ndarray,
                         out_dir: str,
                         fname: str,
                         save_size=(1920, 1080)):
    """Save only the pure heatmap in the same folder as the bbox image."""
    os.makedirs(out_dir, exist_ok=True)

    target_w, target_h = save_size

    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.ndim != 2:
        amap = np.squeeze(amap)

    # Normalize each image map to [0, 1] for visualization.
    amap = amap - np.nanmin(amap)
    denom = np.nanmax(amap) + 1e-8
    amap = amap / denom
    amap_u8 = (amap * 255).clip(0, 255).astype(np.uint8)
    amap_u8 = cv2.resize(amap_u8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # OpenCV colormap is BGR, convert to RGB for PIL saving.
    heatmap_bgr = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    stem, ext = os.path.splitext(fname)
    ext = ext if ext else '.jpg'
    heatmap_pil.save(os.path.join(out_dir, f"{stem}_heatmap{ext}"))

def save_outputs(img_tensor: torch.Tensor,
                 anomaly_map: np.ndarray,
                 out_dir: str,
                 fname: str,
                 threshold: float,
                 min_area: int,
                 save_size=(1920, 1080),
                 save_heatmap: bool = True,
                 original_image_path: Optional[str] = None):
    """Save bbox image and heatmap image together in the same output folder.

    If original_image_path is given, bboxes are drawn on the original unmasked test image.
    The model can use a masked tensor, but visualization remains on the original image.
    """
    os.makedirs(out_dir, exist_ok=True)

    if original_image_path is not None:
        img_pil = Image.open(original_image_path).convert('RGB')
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_u8 = ((img_tensor.cpu() * std + mean).clamp(0, 1) * 255).byte()
        img_pil = Image.fromarray(img_u8.permute(1, 2, 0).numpy())

    bboxes = anomaly_map_to_bboxes(anomaly_map, threshold=threshold, min_area=min_area)

    target_w, target_h = save_size
    resized_img = img_pil.resize((target_w, target_h), Image.BILINEAR)

    # anomaly_map is produced at model input resolution, usually 512x512.
    amap = np.asarray(anomaly_map)
    if amap.ndim != 2:
        amap = np.squeeze(amap)
    map_h, map_w = amap.shape[-2], amap.shape[-1]

    scale_x = target_w / map_w
    scale_y = target_h / map_h
    scaled_bboxes = [
        (int(x0 * scale_x), int(y0 * scale_y), int(x1 * scale_x), int(y1 * scale_y))
        for x0, y0, x1, y1 in bboxes
    ]

    stem, ext = os.path.splitext(fname)
    ext = ext if ext else '.jpg'

    boxed = draw_bboxes_on_image(resized_img, scaled_bboxes, color='red', width=6)
    boxed.save(os.path.join(out_dir, f"{stem}_bbox{ext}"))

    if save_heatmap:
        save_heatmap_outputs(
            anomaly_map=anomaly_map,
            out_dir=out_dir,
            fname=fname,
            save_size=save_size,
        )

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
    c.class_name = 'posco'
    c.posco_train_subdir = None
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

    # Discover folders from checkpoints and abnormal/<folder>.
    # normal/<folder> is optional because some POSCO test setups only visualize abnormal images.
    candidate_sets = []
    for base in [msflow_base, rf_base, abnormal_base]:
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
        apply_test_mask=args.apply_test_mask,
        mask_dir=args.mask_dir,
        mask_threshold=args.mask_threshold,
        save_masked_debug=args.save_masked_test_debug,
        masked_debug_dir=args.masked_test_debug_dir,
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
                save_heatmap=args.save_heatmap,
                original_image_path=img_paths[b],
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

    dataset = PoscoFlatTestDataset(
        args.data_root,
        input_size=cfg.input_size,
        img_mean=cfg.img_mean,
        img_std=cfg.img_std,
        apply_test_mask=args.apply_test_mask,
        mask_dir=args.mask_dir,
        mask_threshold=args.mask_threshold,
        save_masked_debug=args.save_masked_test_debug,
        masked_debug_dir=args.masked_test_debug_dir,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    print(f"[INFO] Found {len(dataset)} images under {args.data_root}")
    extractor, parallel_flows, fusion_flow = build_msflow(cfg, args.msflow_ckpt)

    print('[INFO] Initializing RF model...')
    init_imgs, _, _, _, _ = next(iter(loader))
    init_imgs = init_imgs.to(cfg.device, non_blocking=True)
    with torch.no_grad():
        _, z_fused_list, _ = msflow_forward(cfg, extractor, parallel_flows, fusion_flow, init_imgs, return_pre_fusion=True)
    rf_model = build_rf_from_batch(cfg.device, args.rf_ckpt, z_fused_list, args.rf_tdims, args.rf_depths)

    seen_dirs = set()
    total_processed = 0
    start = time.time()

    for imgs, img_paths, label_names, folder_names, fnames in loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        final_maps = get_final_localization_map(cfg, extractor, parallel_flows, fusion_flow, rf_model, imgs, args.rf_steps)
        total_processed += imgs.shape[0]

        for b in range(imgs.shape[0]):
            final_map = final_maps[b]
            if torch.is_tensor(final_map):
                final_map = final_map.detach().cpu().numpy()

            out_dir = os.path.join(args.output_dir, label_names[b])
            if out_dir not in seen_dirs:
                os.makedirs(out_dir, exist_ok=True)
                seen_dirs.add(out_dir)

            save_outputs(
                imgs[b],
                final_map,
                out_dir,
                fnames[b],
                args.threshold,
                args.min_area,
                save_heatmap=args.save_heatmap,
                original_image_path=img_paths[b],
            )

    fps = total_processed / max(time.time() - start, 1e-6)
    print(datetime.datetime.now().strftime('[%Y-%m-%d-%H:%M:%S]'),
          f'Done. Processed {total_processed} images, FPS: {fps:.1f}')


def main():
    parser = argparse.ArgumentParser(description='Visualize POSCO bounding boxes from MSFlow+RF localization map')
    parser.add_argument('--data_root', type=str, default='./data/posco/test',
                        help='POSCO test root containing normal/*.jpg and abnormal/*.jpg')
    parser.add_argument('--output_dir', type=str, default='./results_bboxes_posco_rf_test',
                        help='Where to save images with bounding boxes')
    parser.add_argument('--save_heatmap', action='store_true', default=True,
                        help='Save pure heatmap image next to bbox image in the same folder. Default: True')
    parser.add_argument('--no_save_heatmap', dest='save_heatmap', action='store_false',
                        help='Disable heatmap saving')
    parser.add_argument('--apply-test-mask', action='store_true', default=False,
                        help='Apply folder-specific mask to a copy of each test image before model inference.')
    parser.add_argument('--mask-dir', type=str, default='./mask',
                        help='Directory containing 02_mask.jpg, 04_mask.jpg, ...')
    parser.add_argument('--mask-threshold', type=int, default=10,
                        help='Pixels <= threshold in mask are treated as black masked-out area.')
    parser.add_argument('--save-masked-test-debug', action='store_true', default=False,
                        help='Save one masked test input image per dataset object for checking.')
    parser.add_argument('--masked-test-debug-dir', type=str, default='./debug_posco_test_mask',
                        help='Directory for masked test debug images.')

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
