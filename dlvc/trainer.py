import collections
import torch
from typing import  Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

from dlvc.wandb_logger import WandBLogger
from dlvc.dataset.oxfordpets import OxfordPetsCustom

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

class ImgSemSegTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5):
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''
        

    
        self.model        = model
        self.optimizer    = optimizer
        self.loss_fn      = loss_fn
        self.lr_scheduler = lr_scheduler
        self.device       = device

        self.num_epochs     = num_epochs
        self.val_frequency  = val_frequency
        self.train_metric   = train_metric
        self.val_metric     = val_metric

        # Oxford-IIIT-Pet masks start at 1, Cityscapes at 0
        self.subtract_one = isinstance(train_data, OxfordPetsCustom)

        # ─── data loaders ───────────────────────────────────────────────
        self.train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True)
        self.val_data_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            num_workers=1, pin_memory=True)

        self.num_train_data = len(train_data)
        self.num_val_data   = len(val_data)

        # ─── checkpoint & optional W&B logger ───────────────────────────
        self.checkpoint_dir = training_save_dir
        self.checkpoint_dir.mkdir(exist_ok=True)

        try:
            from dlvc.wandb_logger import WandBLogger
            self.wandb_logger = WandBLogger(enabled=True,
                                            model=model,
                                            run_name=model.net._get_name())
        except Exception:
            self.wandb_logger = None
        

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """
        self.model.train()
        self.train_metric.reset()
        epoch_loss = 0.0

        for inputs, labels in tqdm(self.train_data_loader,
                                   desc=f"train {epoch_idx}",
                                   total=len(self.train_data_loader)):
            self.optimizer.zero_grad()

            labels = labels.squeeze(1) - int(self.subtract_one)
            batch_size = inputs.size(0)

            outputs = self.model(inputs.to(self.device))
            if isinstance(outputs, collections.OrderedDict):   # torchvision FCN
                outputs = outputs['out']

            loss = self.loss_fn(outputs, labels.to(self.device))
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * batch_size
            self.train_metric.update(outputs.detach().cpu(),
                                     labels.detach().cpu())

        self.lr_scheduler.step()
        epoch_loss /= self.num_train_data
        epoch_miou  = self.train_metric.mIoU()

        print(f"______epoch {epoch_idx}\nLoss: {epoch_loss}")
        print(self.train_metric)

        return epoch_loss, epoch_miou


    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        self.model.eval()
        self.val_metric.reset()
        epoch_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_data_loader,
                                       desc=f"eval {epoch_idx}",
                                       total=len(self.val_data_loader)):
                labels = labels.squeeze(1) - int(self.subtract_one)
                batch_size = inputs.size(0)

                outputs = self.model(inputs.to(self.device))
                if isinstance(outputs, collections.OrderedDict):
                    outputs = outputs['out']

                loss = self.loss_fn(outputs, labels.to(self.device))

                epoch_loss += loss.item() * batch_size
                self.val_metric.update(outputs.cpu(), labels.cpu())

        epoch_loss /= self.num_val_data
        epoch_miou  = self.val_metric.mIoU()

        print(f"______epoch {epoch_idx} - validation\nLoss: {epoch_loss}")
        print(self.val_metric)

        return epoch_loss, epoch_miou

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean IoU on validation data set is higher
        than currently saved best mean IoU or if it is end of training. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        best_miou = 0.0
        for epoch_idx in range(self.num_epochs):

            train_loss, train_miou = self._train_epoch(epoch_idx)
            log_dict = {'epoch': epoch_idx,
                        'train/loss': train_loss,
                        'train/mIoU': train_miou}

            # run validation only at the chosen frequency (or last epoch)
            if epoch_idx % self.val_frequency == 0 or epoch_idx == self.num_epochs - 1:
                val_loss, val_miou = self._val_epoch(epoch_idx)
                log_dict.update({'val/loss': val_loss,
                                 'val/mIoU': val_miou})

                # save best model
                if val_miou >= best_miou:
                    best_miou = val_miou
                    print(f"#### best mIoU so far: {best_miou:.4f}")
                    print(f"#### saving checkpoint → {self.checkpoint_dir}")
                    self.model.save(Path(self.checkpoint_dir), suffix="best")

                # always save final weights
                if epoch_idx == self.num_epochs - 1:
                    self.model.save(Path(self.checkpoint_dir), suffix="last")

            if self.wandb_logger is not None:
                self.wandb_logger.log(log_dict)

    def dispose(self):
        self.wandb_logger.finish()

                





            
            


