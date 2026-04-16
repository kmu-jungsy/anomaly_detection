import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

__all__ = ('MVTecDataset', 'VisADataset', 'ShanghaiTechDataset', 'POSCODataset')

MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        x = Image.open(x)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

VISA_CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 
                    'fryum', 'macaroni1', 'macaroni2', 
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

class VisADataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)



POSCO_CLASS_NAMES = ['posco']

class POSCODataset(Dataset):
    """
    Expected folder structure:
      <data_path>/
        train/
          *.jpg|png|jpeg
        test/
          normal/*.jpg|png|jpeg
          abnormal/*.jpg|png|jpeg

    Behavior:
      - train: uses only images in train/
      - test:  uses test/normal as label 0, test/abnormal as label 1
      - no pixel masks are available, so mask is always zeros
    """
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.is_train = is_train
        self.input_size = c.input_size
        self.x, self.y = self.load_dataset_folder()

        self.transform_x = T.Compose([
            T.Resize(c.input_size, InterpolationMode.LANCZOS),
            T.ToTensor()
        ])
        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_path = self.x[idx]
        y = int(self.y[idx])

        x = Image.open(x_path).convert('RGB')
        x = self.normalize(self.transform_x(x))

        # no pixel-level mask in POSCO
        mask = torch.zeros([1, *self.input_size], dtype=torch.float32)

        return x, y, mask

    def load_dataset_folder(self):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

        if self.is_train:
            train_dir = os.path.join(self.dataset_path, 'train')
            assert os.path.isdir(train_dir), f"Missing train folder: {train_dir}"

            train_paths = sorted([
                os.path.join(train_dir, f)
                for f in os.listdir(train_dir)
                if f.lower().endswith(exts)
            ])

            assert len(train_paths) > 0, f"No training images found in: {train_dir}"

            x = train_paths
            y = [0] * len(train_paths)   # all training images are normal
            return x, y

        else:
            test_normal_dir = os.path.join(self.dataset_path, 'test', 'normal')
            test_abnormal_dir = os.path.join(self.dataset_path, 'test', 'abnormal')

            assert os.path.isdir(test_normal_dir), f"Missing test normal folder: {test_normal_dir}"
            assert os.path.isdir(test_abnormal_dir), f"Missing test abnormal folder: {test_abnormal_dir}"

            normal_paths = sorted([
                os.path.join(test_normal_dir, f)
                for f in os.listdir(test_normal_dir)
                if f.lower().endswith(exts)
            ])

            abnormal_paths = sorted([
                os.path.join(test_abnormal_dir, f)
                for f in os.listdir(test_abnormal_dir)
                if f.lower().endswith(exts)
            ])

            assert len(normal_paths) + len(abnormal_paths) > 0, \
                f"No test images found in: {os.path.join(self.dataset_path, 'test')}"

            x = normal_paths + abnormal_paths
            y = [0] * len(normal_paths) + [1] * len(abnormal_paths)
            return x, y


# --------------------------- ShanghaiTech (frame-level) ---------------------------
# Expected folder structure:
#   <data_path>/
#     training/frames/<video_id>/*.jpg
#     testing/frames/<video_id>/*.jpg
#     testing/test_frame_mask/<video_id>.npy
#
# We do **frame-level** anomaly detection:
#   - train: all frames are normal (label=0)
#   - test: label is loaded from .npy (per-frame), or inferred from per-pixel masks

class ShanghaiTechDataset(Dataset):
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.is_train = is_train
        self.input_size = c.input_size

        self.x, self.y = self.load_dataset_folder()

        self.transform_x = T.Compose([
            T.Resize(c.input_size, InterpolationMode.LANCZOS),
            T.ToTensor(),
        ])
        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_path = self.x[idx]
        y = self.y[idx]
        x = Image.open(x_path).convert('RGB')
        x = self.normalize(self.transform_x(x))
        # keep signature compatible with existing training loop: (image, label, mask)
        mask = torch.zeros([1, *self.input_size], dtype=torch.float32)
        return x, int(y), mask

    @staticmethod
    def _frame_label_from_mask(mask_arr, i: int) -> int:
        """Convert a ShanghaiTech mask array into a frame label (0/1)."""
        # common cases:
        #  (T,) or (T,1) => already frame-level
        #  (T,H,W) or (T,1,H,W) => pixel-level per frame
        m = mask_arr
        if m.ndim == 1:
            return int(m[i] > 0)
        if m.ndim == 2 and m.shape[1] == 1:
            return int(m[i, 0] > 0)
        # pixel-wise
        if m.ndim == 3:
            return int(np.any(m[i] > 0))
        if m.ndim == 4 and m.shape[1] == 1:
            return int(np.any(m[i, 0] > 0))
        # fallback: try flattening per-frame
        return int(np.any(np.asarray(m[i]).reshape(-1) > 0))

    def load_dataset_folder(self):
        phase = 'training' if self.is_train else 'testing'
        frames_root = os.path.join(self.dataset_path, phase, 'frames')
        assert os.path.isdir(frames_root), f"Missing frames folder: {frames_root}"

        x, y = [], []
        video_ids = sorted([d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))])

        if self.is_train:
            for vid in video_ids:
                vid_dir = os.path.join(frames_root, vid)
                frame_paths = sorted(
                    [os.path.join(vid_dir, f) for f in os.listdir(vid_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                )
                x.extend(frame_paths)
                y.extend([0] * len(frame_paths))
            return x, y

        # testing: load per-video labels from test_frame_mask
        mask_root = os.path.join(self.dataset_path, 'testing', 'test_frame_mask')
        assert os.path.isdir(mask_root), f"Missing test_frame_mask folder: {mask_root}"

        for vid in video_ids:
            vid_dir = os.path.join(frames_root, vid)
            frame_paths = sorted(
                [os.path.join(vid_dir, f) for f in os.listdir(vid_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            )

            mask_path = os.path.join(mask_root, f"{vid}.npy")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Missing mask file for video '{vid}': {mask_path}")
            mask_arr = np.load(mask_path)

            T_frames = len(frame_paths)
            T_mask = int(mask_arr.shape[0])

            # tolerate off-by-one in some dumps
            if T_mask == T_frames + 1:
                mask_arr = mask_arr[1:]
                T_mask = int(mask_arr.shape[0])
            elif T_mask == T_frames - 1:
                frame_paths = frame_paths[1:]
                T_frames = len(frame_paths)

            if T_mask != T_frames:
                raise ValueError(
                    f"Frame/mask length mismatch for '{vid}': frames={T_frames}, mask={T_mask}. "
                    "Check sorting/naming or the .npy format."
                )

            for i, fp in enumerate(frame_paths):
                x.append(fp)
                y.append(self._frame_label_from_mask(mask_arr, i))

        return x, y