import os
import math
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import instantiate_class, LightningCLI
from torchmetrics import MetricCollection, Accuracy

from src import CIFAR, LabelSmoothing, CosineLR, cutmix, cutout, mixup, get_efficientnet_v2_hyperparam


class BaseVisionSystem(LightningModule):
    def __init__(self, model_name: str, pl: bool, model_args: dict, num_classes: int, num_step: int,
                 gpus: str, max_epochs: int, optimizer_init: dict, lr_scheduler_init: dict, augmentation: str):
        """ Define base vision classification system
        :arg
            backbone_init: feature extractor
            num_classes: number of class of dataset
            num_step: number of step
            gpus: gpus id
            max_epoch: max number of epoch
            optimizer_init: optimizer class path and init args
            lr_scheduler_init: learning rate scheduler class path and init args
        """
        super(BaseVisionSystem, self).__init__()

        # step 1. save data related info (not defined here)
        self.augmentation = augmentation
        self.gpus = len(gpus.split(',')) - 1
        self.num_step = int(math.ceil(num_step / (self.gpus)))
        self.max_epochs = max_epochs

        # step 2. define model
        self.backbone = torch.hub.load('hankyul2/EfficientNetV2-pytorch', model_name, nclass=0, skip_validation=True, **model_args)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

        # step 2.2. progressive learning (pl)
        self.use_pl = pl
        self.stage_param, self.eval_size = get_efficientnet_v2_hyperparam(model_name)
        self.mixup_ratio = None

        # step 3. define lr tools (optimizer, lr scheduler)
        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init
        self.criterion = LabelSmoothing()

        # step 4. define metric
        metrics = MetricCollection({'top@1': Accuracy(top_k=1), 'top@5': Accuracy(top_k=5)})
        self.train_metric = metrics.clone(prefix='train/')
        self.valid_metric = metrics.clone(prefix='valid/')
        self.test_metric = metrics.clone(prefix='test/')

    def forward(self, batch, batch_idx):
        x, y = batch
        loss, y_hat = self.compute_loss_eval(x, y)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.use_pl:
            cur_stage_idx = min(3, int(4 * (self.current_epoch+1) / self.max_epochs))
            batch_size, train_size, dropout, randaug, mixup = self.stage_param[cur_stage_idx]
            self.trainer.datamodule.batch_size = int(batch_size)
            self.trainer.datamodule.size = int(train_size)
            self.trainer.datamodule.eval_size = int(self.eval_size)
            self.trainer.datamodule.update_transforms()
            self.trainer.datamodule.train_ds.dataset.transform = self.trainer.datamodule.train_transform
            self.backbone.change_dropout_rate(dropout)
            self.mixup_ratio = mixup

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.shared_step(batch, self.train_metric, 'train', add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.valid_metric, 'valid', add_dataloader_idx=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.test_metric, 'test', add_dataloader_idx=True)

    def shared_step(self, batch, metric, mode, add_dataloader_idx):
        x, y = batch
        loss, y_hat = self.compute_loss(x, y) if mode == 'train' else self.compute_loss_eval(x, y)
        metric = metric(y_hat, y)
        self.log_dict({f'{mode}/loss': loss}, add_dataloader_idx=add_dataloader_idx, prog_bar=True)
        self.log_dict(metric, add_dataloader_idx=add_dataloader_idx, prog_bar=True)
        return loss

    def compute_loss(self, x, y):
        if self.augmentation == 'default':
            return self.compute_loss_eval(x, y)

        elif self.augmentation == 'mixup':
            x, y1, y2, ratio = mixup(x, y, self.mixup_ratio)
            y_hat = self.fc(self.backbone(x))
            loss = self.criterion(y_hat, y1) * ratio + self.criterion(y_hat, y2) * (1 - ratio)
            return loss, y_hat

        elif self.augmentation == 'cutout':
            x, y, ratio = cutout(x, y)
            return self.compute_loss_eval(x, y)

        elif self.augmentation == 'cutmix':
            x, y1, y2, ratio = cutmix(x, y)
            y_hat = self.fc(self.backbone(x))
            loss = self.criterion(y_hat, y1) * ratio + self.criterion(y_hat, y2) * (1 - ratio)
            return loss, y_hat

    def compute_loss_eval(self, x, y):
        y_hat = self.fc(self.backbone(x))
        loss = self.criterion(y_hat, y)
        return loss, y_hat

    def configure_optimizers(self):
        optimizer = instantiate_class(self.parameters(), self.optimizer_init_config)

        lr_scheduler = {'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
                        'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def update_and_get_lr_scheduler_config(self):
        if 'num_step' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['num_step'] = self.num_step
        if 'max_epochs' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['max_epochs'] = self.max_epochs
        if 'max_lr' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['max_lr'] = self.optimizer_init_config['init_args']['lr']
        if 'total_steps' in self.lr_scheduler_init_config['init_args']:
            if self.use_pl:
                total_batch = [param[0] for param in self.stage_param for _ in range(self.max_epochs // 4)]
                total_steps = sum(int(math.ceil(self.trainer.datamodule.data_len / step / self.gpus)) for step in total_batch)
                self.lr_scheduler_init_config['init_args']['total_steps'] = total_steps
            else:
                self.lr_scheduler_init_config['init_args']['total_steps'] = self.num_step * self.max_epochs
        return self.lr_scheduler_init_config

    @property
    def lr(self):
        return self.optimizer_init_config['init_args']['lr']

    @lr.setter
    def lr(self, val):
        self.optimizer_init_config['init_args']['lr'] = val


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 1. link argument
        parser.link_arguments('data.num_classes', 'model.num_classes', apply_on='instantiate')
        parser.link_arguments('data.num_step', 'model.num_step', apply_on='instantiate')
        parser.link_arguments('trainer.max_epochs', 'model.max_epochs', apply_on='parse')
        parser.link_arguments('trainer.gpus', 'model.gpus', apply_on='parse')

        # 2. add optimizer & scheduler argument
        parser.add_optimizer_args((SGD, Adam, AdamW), link_to='model.optimizer_init')
        parser.add_lr_scheduler_args((CosineLR, OneCycleLR), link_to='model.lr_scheduler_init')


if __name__ == '__main__':
    cli = MyLightningCLI(BaseVisionSystem, CIFAR, save_config_overwrite=True)
    cli.trainer.test(ckpt_path='best', dataloaders=cli.datamodule.test_dataloader())