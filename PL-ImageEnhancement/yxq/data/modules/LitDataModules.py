import lightning as L
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities import CombinedLoader
from typing import Any, Dict, Generator, Iterable, List, Optional, Union, Callable
from ..utils import PairedImageDataset, SingleImageDataset

from torchvision.datasets import  ImageFolder, Flickr30k, MNIST
from torchvision import transforms

LitDataArgs = Union[Dict[str, Any], List[Dict[str, Any]]]

class IRLitDataModule(L.LightningDataModule):
    def __init__(
        self,
        train: Optional[LitDataArgs]=None,
        val: Optional[LitDataArgs]=None,
        test: Optional[LitDataArgs]=None,
        predict: Optional[LitDataArgs]=None
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.predict = predict

    def setup(self, stage=None):
        if stage == "fit":
            self.train_sets = self.get_datasets(PairedImageDataset, self.train)
            self.val_sets = self.get_datasets(PairedImageDataset, self.val)
        if stage == "validate":
            self.val_sets = self.get_datasets(PairedImageDataset, self.val)
        if stage == "test":
            self.test_sets = self.get_datasets(PairedImageDataset, self.test)
        if stage == "predict":
            self.predict_sets = self.get_datasets(SingleImageDataset, self.predict)

    def train_dataloader(self):
        return self.get_dataloaders(self.train_sets, self.train)

    def val_dataloader(self):
        return self.get_dataloaders(self.val_sets, self.val)

    def test_dataloader(self):
        return self.get_dataloaders(self.test_sets, self.test)

    def predict_dataloader(self):
        return self.get_dataloaders(self.predict_sets, self.predict)
    
    def get_datasets(
            self, 
            Dataset: Callable[[Dict], Dataset], 
            data_args: LitDataArgs
            ) -> Union[Dataset, List[Dataset]]:
        if isinstance(data_args, list):
            return [Dataset(**set_args['dataset']) for set_args in data_args]
        else:
            return Dataset(**data_args['dataset'])

    def get_dataloaders(
            self, 
            datasets: Union[Dataset, List[Dataset]], 
            dataloader_args: LitDataArgs
            ) -> Union[DataLoader, List[DataLoader]]:
        if isinstance(datasets, list):
            return [DataLoader(dataset, **loader_args['dataloader']) for dataset, loader_args in zip(datasets, dataloader_args)]
        else:
            return DataLoader(datasets, **dataloader_args['dataloader'])