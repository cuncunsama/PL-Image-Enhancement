import torch
from torch.nn import Module
import torch.nn.functional as F
import lightning as L

from typing import Any, Dict, Generator, Iterable, List, Optional, Union

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from torchmetrics.metric import Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable, ArgsType

# for prediction
from torchvision.utils import save_image
import os

import scipy.io
import numpy as np
import pandas as pd
import base64

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

class IRLitModule(L.LightningModule):
    def __init__(
        self,
        net: Module,
        losses: Union[Module, List[Module]],
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
        lr_scheduler_config: ArgsType,
    ):
        super().__init__()
        self.model = net
        self.example_input_array = torch.rand(8, 3, 256, 256)
        # loss
        if not isinstance(losses, list):
            losses = [losses]
        self.losses = losses
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr_scheduler_config = lr_scheduler_config

        torch.set_float32_matmul_precision("medium")
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        if not isinstance(y_hat, list):
            y_hat = [y_hat]
        # loss
        train_loss = 0
        for loss in self.losses:
            for y_pred in y_hat:
                train_loss += loss(torch.clamp(y_pred, 0, 1), y)
        # in training_step, default: on_step=True, on_epoch=False
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)

        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = self.check_img_size(x, 16)
        y = self.check_img_size(y, 16)
        y_hat = self.model(x)
        if not isinstance(y_hat, list):
            y_hat = [y_hat]
        # loss
        val_loss = 0
        for loss in self.losses:
            for y_pred in y_hat:
                val_loss += loss(torch.clamp(y_pred, 0, 1), y)
        # metrics

        if len(y_hat) == 1:
            val_psnr = calculate_psnr(y, y_hat[0], 0)
            val_ssim = calculate_ssim(y, y_hat[0], 0)
            val_metrics = {"val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim}
        else:
            val_psnr = calculate_psnr(y, y_hat[0], 0)
            val_ssim = calculate_ssim(y, y_hat[0], 0)
            val_psnr1 = calculate_psnr(y, y_hat[1], 0)
            val_ssim1 = calculate_ssim(y, y_hat[1], 0)
            val_metrics = {"val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim, "val_psnr1": val_psnr1, "val_ssim1": val_ssim1}

        # in validation_step, default: on_step=False, on_epoch=True
        self.log_dict(val_metrics, sync_dist=True, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        # For MPRNet, the input size should be multiple of 8
        # For HINet, the input size should be multiple of 16
        x = self.check_img_size(x, 16)
        y = self.check_img_size(y, 16)

        y_hat = self.model(x)
        if not isinstance(y_hat, list):
            y_hat = [y_hat]

        # loss
        test_loss = 0
        for loss in self.losses:
            for y_pred in y_hat:
                test_loss += loss(torch.clamp(y_pred, 0, 1), y)
        # metrics
        test_psnr = calculate_psnr(y, y_hat[0], 0)
        test_ssim = calculate_ssim(y, y_hat[0], 0)
        # print(f'test_psnr = {test_psnr}, test_ssim = {test_ssim}')

        test_metrics = {"test_loss": test_loss, "test_psnr": test_psnr, "test_ssim": test_ssim}
        # in test_step, default: on_step=False, on_epoch=True
        self.log_dict(test_metrics)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self.check_img_size(batch, 16)
        # y_hat = self.model(batch)
        output_folder = r'C:\Users\yxq\datasets\SIDD\predictions\test0'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for i, img in enumerate(batch):
            save_image(img, os.path.join(output_folder, f'{batch_idx}-{i}.png'))
        
    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters())
        lr_scheduler = self.scheduler(optimizer)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            **self.lr_scheduler_config,
        }
        return [optimizer], [lr_scheduler_config]

    def check_img_size(self, x, div_size=16):
        _, _, h, w = x.size()
        mod_pad_h = (div_size - h % div_size) % div_size
        mod_pad_w = (div_size - w % div_size) % div_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
