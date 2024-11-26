from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

# trainer
from yxq.callback import *
from lightning.pytorch.callbacks import *

# model
from yxq.model import IRLitModule
from yxq.model.arch import NAFNet, CascadedGaze, KBNet_s, Restomer, HINet, MPRNet, HAINet, TestNet, TestHI
from yxq.model.arch import HINetLocal, MPRNetLocal
from yxq.model.loss import PSNRLoss, L1Loss, MSELoss, CharbonnierLoss, MultiHeadPSNRLoss
# from torch.nn.functional import l1_loss, mse_loss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# optimizer
from torch.optim import AdamW, Adam
from lightning.pytorch.cli import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# data
from yxq.data import IRLitDataModule
from torchvision.datasets import Flickr30k


from lightning.pytorch.strategies import DDPStrategy


class IRLitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        pass


def cli_main():
    trainer_defaults = {
        "callbacks": [
            LearningRateMonitor(logging_interval="step"),
            # FineTuneLearningRateFinder(milestones=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            # ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_weights_only=False, filename="{epoch}-{train_loss:.2f}"),
            ModelCheckpoint(monitor="val_ssim", mode="max", save_top_k=3, save_weights_only=False, filename="{epoch}-{val_ssim:.4f}"),
            ModelCheckpoint(monitor="val_psnr", mode="max", save_top_k=1, save_weights_only=False, filename="{epoch}-{val_psnr:.4f}"),
            ModelSummary(max_depth=1),
        ],
    }

    cli = IRLitCLI(trainer_defaults=trainer_defaults)


if __name__ == "__main__":
    cli_main()
