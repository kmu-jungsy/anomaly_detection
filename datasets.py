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
        self.train_subdir = getattr(c, 'posco_train_subdir', None)

        # POSCO ROI mask option.
        # Example:
        #   data/posco/train/02/*.jpg uses <posco_mask_dir>/02_mask.jpg
        # Mask rule:
        #   black pixels in mask -> make image black
        #   white pixels in mask -> keep original image
        self.apply_train_mask = bool(getattr(c, 'posco_apply_train_mask', False)) and self.is_train
        self.mask_dir = getattr(c, 'posco_mask_dir', './mask')
        self.mask_threshold = int(getattr(c, 'posco_mask_threshold', 10))
        self._mask_cache = {}
        self.save_train_mask_debug = bool(getattr(c, 'posco_save_train_mask_debug', False)) and self.apply_train_mask
        self.mask_debug_dir = getattr(c, 'posco_mask_debug_dir', './debug_posco_train_mask')

        self.x, self.y = self.load_dataset_folder()

        self.transform_x = T.Compose([
            T.Resize(c.input_size, InterpolationMode.LANCZOS),
            T.ToTensor()
        ])
        self.transform_roi_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()
        ])
        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

        # Optional: save one masked training image before training starts.
        # This helps verify whether the POSCO ROI mask is applied correctly.
        if self.save_train_mask_debug:
            self._save_one_masked_train_debug_image()

    def __len__(self):
        return len(self.x)

    def _get_train_folder_name(self, image_path):
        """Return POSCO train subfolder name such as '02', '04', ..."""
        if self.train_subdir:
            return self.train_subdir

        train_root = os.path.join(self.dataset_path, 'train')
        rel_path = os.path.relpath(image_path, train_root)
        folder_name = rel_path.split(os.sep)[0]
        return folder_name

    def _load_roi_mask(self, folder_name):
        """Load and cache binary ROI mask for a POSCO train folder.

        Returned tensor shape: [1, H, W]
          1.0 = keep original image
          0.0 = black out image
        """
        if folder_name in self._mask_cache:
            return self._mask_cache[folder_name]

        mask_path = os.path.join(self.mask_dir, f'{folder_name}_mask.jpg')
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f'Missing POSCO mask file for folder {folder_name}: {mask_path}'
            )

        mask_img = Image.open(mask_path).convert('L')
        mask_tensor = self.transform_roi_mask(mask_img)

        # White area remains 1, black area becomes 0.
        keep_mask = (mask_tensor > (self.mask_threshold / 255.0)).float()
        self._mask_cache[folder_name] = keep_mask
        return keep_mask

    def _save_one_masked_train_debug_image(self):
        """Save one masked training sample for visual debugging.

        The saved image is BEFORE normalization, so it should look like a normal image:
          - black mask area: black
          - white mask area: original image
        """
        if len(self.x) == 0:
            return

        os.makedirs(self.mask_debug_dir, exist_ok=True)

        x_path = self.x[0]
        folder_name = self._get_train_folder_name(x_path)

        x_img = Image.open(x_path).convert('RGB')
        x = self.transform_x(x_img)
        keep_mask = self._load_roi_mask(folder_name)
        masked_x = x * keep_mask

        original_stem = os.path.splitext(os.path.basename(x_path))[0]
        save_path = os.path.join(
            self.mask_debug_dir,
            f'{folder_name}_masked_debug_{original_stem}.jpg'
        )

        # masked_x is in [0, 1]. Convert to PIL and save as jpg.
        T.ToPILImage()(masked_x.clamp(0, 1)).save(save_path)
        print(f'[POSCO mask debug] saved masked training sample: {save_path}')

    def __getitem__(self, idx):
        x_path = self.x[idx]
        y = int(self.y[idx])

        x_img = Image.open(x_path).convert('RGB')
        x = self.transform_x(x_img)

        # Apply ROI mask only to POSCO training images.
        # black mask area -> black image pixels, white mask area -> original pixels
        if self.apply_train_mask:
            folder_name = self._get_train_folder_name(x_path)
            keep_mask = self._load_roi_mask(folder_name)
            x = x * keep_mask

        x = self.normalize(x)

        # no pixel-level anomaly mask in POSCO
        mask = torch.zeros([1, *self.input_size], dtype=torch.float32)

        return x, y, mask

    def load_dataset_folder(self):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

        def collect_images(root_dir):
            """Collect image files recursively under root_dir."""
            image_paths = []
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in filenames:
                    if fname.lower().endswith(exts):
                        image_paths.append(os.path.join(dirpath, fname))
            return sorted(image_paths)

        if self.is_train:
            train_dir = os.path.join(self.dataset_path, 'train')

            # When main.py sets self.train_subdir, train only on that folder.
            # Example: data/posco/train/02/*.jpg
            # Otherwise, train on the whole POSCO train set recursively:
            # Example: data/posco/train/01/*.jpg + ... + data/posco/train/10/*.jpg
            if self.train_subdir:
                train_dir = os.path.join(train_dir, self.train_subdir)

            assert os.path.isdir(train_dir), f"Missing train folder: {train_dir}"

            train_paths = collect_images(train_dir)

            assert len(train_paths) > 0, f"No training images found in: {train_dir}"
            print(f"POSCO train folder: {train_dir} ({len(train_paths)} images, recursive=True)")

            x = train_paths
            y = [0] * len(train_paths)   # all training images are normal
            return x, y
        else:
            test_normal_dir = os.path.join(self.dataset_path, 'test', 'normal')
            test_abnormal_dir = os.path.join(self.dataset_path, 'test', 'abnormal')

            assert os.path.isdir(test_normal_dir), f"Missing test normal folder: {test_normal_dir}"
            assert os.path.isdir(test_abnormal_dir), f"Missing test abnormal folder: {test_abnormal_dir}"

            # Supports both:
            #   data/posco/test/normal/*.jpg
            #   data/posco/test/normal/01/*.jpg, data/posco/test/normal/02/*.jpg, ...
            normal_paths = collect_images(test_normal_dir)
            abnormal_paths = collect_images(test_abnormal_dir)

            assert len(normal_paths) + len(abnormal_paths) > 0, \
                f"No test images found in: {os.path.join(self.dataset_path, 'test')}"

            x = normal_paths + abnormal_paths
            y = [0] * len(normal_paths) + [1] * len(abnormal_paths)
            print(
                f"POSCO test folders: normal={len(normal_paths)} images, "
                f"abnormal={len(abnormal_paths)} images, recursive=True"
              
            )
            return x, y
