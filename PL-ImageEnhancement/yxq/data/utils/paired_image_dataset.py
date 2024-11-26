# yxq.data.utils
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

from .file_client import FileClient
from .get_paths import paired_paths_from_lmdb, paired_paths_from_folder, paths_from_lmdb, paths_from_folder

from .img_util import imfrombytes, img2tensor, padding
from .transform import augment

import numpy as np
from PIL import Image
import litdata as ld
from litdata import StreamingDataset, StreamingDataLoader



class SingleImageDataset(Dataset):
    def __init__(
            self, 
            flag: int,
            dir: str,
            patch_size: Optional[int]=None,
            use_flip: bool=False, 
            use_rot: bool=False, 
            sigma: Optional[int]=None, 
            ):
        super(SingleImageDataset, self).__init__()

        self.flag = flag
        io_backend = {}
        if dir.endswith(".lmdb"):
            io_backend['backend'] = 'lmdb'
            io_backend['db_paths'] = dir
            self.paths = paths_from_lmdb(dir)
        else:
            self.paths = paths_from_folder(dir)

        self.file_client = FileClient(**io_backend)

        # data augmentations
        self.patch_size = patch_size
        self.use_flip, self.use_rot = use_flip, use_rot
        self.sigma = sigma if sigma is not None else None

    def __getitem__(self, index):
        # get image
        path = self.paths[index]
        img_bytes = self.file_client.get(path)
        try:
            img = imfrombytes(img_bytes, self.flag, float32=True)
        except:
            raise Exception("path {} not working".format(path))
        
        # augmentation
        if self.patch_size is not None:
            img = padding(img, self.patch_size)   
        img= augment(img, self.use_flip, self.use_rot)
        # result
        img = img2tensor(img, bgr2rgb=True, float32=True)

        if self.sigma is not None:
            noise_level = torch.FloatTensor([self.sigma])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img.size()).mul_(noise_level).float()
            img.add_(noise)

        return img

    def __len__(self):
        return len(self.paths)

class PairedImageDataset(Dataset):
    def __init__(
            self, 
            flag: int,
            lq_dir: str,
            gt_dir: str,
            patch_size: int | List | None = None,
            use_flip: bool=False, 
            use_rot: bool=False,
            sigma: Optional[int]=None, 
            ):
        ''' paired image dataset for image restoration
            Args:
                flag: 0 for grayscale, 1 for color, -1 for unchanged
                lq_dir: path to lq and gt images, can be either a folder or a lmdb database
                gt_dir: path to lq and gt images, can be either a folder or a lmdb database
                patch_size: if not None, crop the image to the patch size
        '''
        super(PairedImageDataset, self).__init__()

        self.flag = flag
        io_backend = {}
        if (lq_dir.endswith(".lmdb") and gt_dir.endswith(".lmdb")):
            io_backend['backend'] = 'lmdb'
            io_backend['db_paths'] = [lq_dir, gt_dir]
            io_backend['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([lq_dir, gt_dir])
        else:
            self.paths = paired_paths_from_folder([lq_dir, gt_dir])

        # tag lmdb_databases with key {lq, gt}, to get lq or gt image from right database
        self.file_client = FileClient(**io_backend)

        # data augmentations
        self.patch_size = patch_size
        self.use_flip, self.use_rot = use_flip, use_rot
        self.sigma = sigma if sigma is not None else None

    def __getitem__(self, index):
        # get lq
        lq_path, gt_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, self.flag, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        # get gt
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, self.flag, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))
        
        # augmentation
        if self.patch_size is not None:
            img_lq, img_gt = padding([img_lq, img_gt], self.patch_size)
        img_lq, img_gt = augment([img_lq, img_gt], self.use_flip, self.use_rot)
        # result
        img_lq, img_gt = img2tensor([img_lq, img_gt], bgr2rgb=True, float32=True)

        if self.sigma is not None:
            noise_level = torch.FloatTensor([self.sigma])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        return img_lq, img_gt

    def __len__(self):
        return len(self.paths)


