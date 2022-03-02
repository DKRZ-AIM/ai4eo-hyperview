#! /usr/bin/env python3

import os
try:
    import nni
except ImportError:
    pass
import copy
import time
import argparse
import numpy as np
import datetime
from collections import defaultdict
import pprint

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.model_summary import ModelSummary

from typing import Optional

class HyperviewDataModule(pl.LightningDataModule):
    """ Lightning data module for Hyperview data """

    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self, args):
        '''Use this method to do things that might write to disk or that need to be done 
        only from a single process in distributed settings.'''
        pass

    def setup(self, stage: Optional[str] = None):
        '''
        Data operations performed on every GPU

        setup() expects an stage: Optional[str] argument. It is used to separate setup logic 
        for trainer.{fit,validate,test}. If setup is called with stage = None, we assume all 
        stages have been set-up.

        Creates self.{train_data, valid_data, test_data} depending on 'stage' (HyperviewDataset)
        '''

        if stage in (None, 'fit'):
            self.train_data = HyperviewDataset('train', self.args)
            self.valid_data = HyperviewDataset('valid', self.args)
        if stage in (None, 'test'):
            self.test_data = HyperviewDataset('test', self.args)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.args.test_batch_size, 
                          num_workers=self.args.num_workers, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.test_batch_size, 
                          num_workers=self.args.num_workers, shuffle=False, drop_last=False)

    def predict_dataloader(self, args): # predicts on test set
        return DataLoader(self.test_data, batch_size=self.args.test_batch_size, 
                          num_workers=self.args.num_workers, shuffle=False, drop_last=False)

    @staticmethod
    def add_dataloader_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num-workers', type=int, default=1, help='dataloader processes')
        return parser

class HyperviewDataset(Dataset):
    """ Handles everything all Datasets of the different Model have in common like loading the same data files."""
    def __init__(self, flag, args):
        '''
        Load data and apply transforms during setup

        Parameters:
        -----------
        flag : string
            Any of train / valid / test. Defines dataset.
        args : argparse Namespace
            arguments passed to the main script
        -----------
        Returns: dataset
        '''
        self.args=args
        
        # TODO
        # load data and labels
        # perform transforms
        # perform augmentation
        # etcpp
        # Ultimately assign self.X (features), self.y (labels)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return (X, y)

class DenseNet(pl.LightningModule):
    def __init__(self, args, input_shapes):
        super().__init__()
        self.args = args
        # model set up goes here
        self.activation = HyperviewNet.activation_fn(self.args.activation)
        self.loss = HyperviewNet.loss_fn(args.loss)

        n_input_values = ... # TODO

        self.fc1 = torch.nn.Linear(n_input_values, self.args.units_dense1)
        self.dr_fc1 = torch.nn.Dropout(self.args.dropout_dense1)
        self.fc2 = torch.nn.Linear(self.args.units_dense1, self.args.units_dense2)
        self.dr_fc2 = torch.nn.Dropout(self.args.dropout_dense2)
        self.fc_final = torch.nn.Linear(self.args.units_dense2, n_output_values)

    def forward(self, x):
        x = self.dr_fc1(self.activation(self.fc1(x)))
        x = self.dr_fc2(self.activation(self.fc2(x)))
        x = self.fc_final(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch-size', type=int, default=256)
        parser.add_argument('--activation', type=str, default='relu')
        parser.add_argument('--units-dense1', type=int, default=64)
        parser.add_argument('--units-dense2', type=int, default=64)
        parser.add_argument('--dropout-dense1', type=float, default=0.0)
        parser.add_argument('--dropout-dense2', type=float, default=0.0)
        return parser

class HyperviewNet(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.save_hyperparameters(self.backbone.args)
        self.best_loss = np.inf # reported for nni
        self.best_epoch = 0

    def forward(self, x):
        y = self.backbone(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x)
        y_pred = torch.squeeze(y_pred, dim=1)
        loss = self.backbone.loss(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x)
        y_pred = torch.squeeze(y_pred, dim=1)
        loss = self.backbone.loss(y_pred, y)
        self.log('valid_loss', loss, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs):
        val_loss = self.trainer.callback_metrics["valid_loss"]
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = self.trainer.current_epoch
        self.log('best_loss', self.best_loss)
        self.log('best_epoch', self.best_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x)
        y_pred = torch.squeeze(y_pred, dim=1)
        loss = self.backbone.loss(y_pred, y)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x)
        y_pred = torch.squeeze(y_pred, dim=1)
        return y_pred

    def configure_optimizers(self):
        if self.backbone.args.optimizer=='adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.backbone.args.optimizer=='sgd':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

    def configure_callbacks(self):
        '''Model checkpoint callback goes here'''
        callbacks = [ModelCheckpoint(monitor='valid_loss', mode='min',
                     dirpath=os.path.join(os.path.dirname(self.backbone.args.save_model_path), 'checkpoint'),
                     filename="hyperviewnet-{epoch}")]
        return callbacks

    @staticmethod
    def activation_fn(activation_name):
        if activation_name == 'tanh':
            return torch.tanh
        elif activation_name == 'relu':
            return F.relu
        elif activation_name == 'sigmoid':
            return torch.sigmoid
        elif activation_name == 'leaky_relu':
            return F.leaky_relu

    @staticmethod
    def loss_fn(loss_name):
        if loss_name == 'mse':
            return F.mse_loss
        elif loss_name == 'mae':
            return F.l1_loss

    @staticmethod
    def best_checkpoint_path(save_model_path, best_epoch):
        '''Path to best checkpoint'''
        ckpt_path = os.path.join(os.path.dirname(save_model_path), 'checkpoint', f"hyperviewnet-epoch={best_epoch}.ckpt")
        all_ckpts = os.listdir(os.path.dirname(ckpt_path))
        # If the checkpoint already exists, lightning creates "*-v1.ckpt"
        only_ckpt = ~np.any([f'-v{best_epoch}' in ckpt for ckpt in all_ckpts])
        assert only_ckpt, f'Cannot load checkpoint: found versioned checkpoints for best_epoch {best_epoch} in {os.path.dirname(ckpt_path)}'
        return ckpt_path

class HyperviewMetricCallbacks(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_validation_epoch_end(self, trainer, pl_module):
        '''After each epoch metrics on validation set'''
        if self.args.nni:
            nni.report_intermediate_result(float(trainer.callback_metrics['valid_loss']))
        metrics = trainer.callback_metrics # everything that was logged in self.log
        epoch = trainer.current_epoch
        print(f'Epoch {epoch} metrics:')
        for key, item in metrics.items():
            print(f'  {key}: {item:.4f}')

    def on_train_epoch_start(self, trainer, pl_module):
        print(f'\nEpoch {trainer.current_epoch} starts training ...')
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        tt = time.time() - self.epoch_start_time
        print(f'Epoch {trainer.current_epoch} finished training in {tt:.0f} seconds')

    def on_test_epoch_end(self, trainer, pl_module):
        pass

    def on_epoch_end(self, trainer, pl_module):
        '''After each epoch (T+V)'''
        pass

    def on_train_end(self, trainer, pl_module):
        '''Final metrics on validation set (after training is done)'''
        print(f'Finished training in {trainer.current_epoch+1} epochs')
        if self.args.nni:
            nni.report_final_result(float(trainer.callback_metrics['best_loss']))

    @staticmethod
    def add_nni_params(args):
        args_nni = nni.get_next_parameter()
        assert all([key in args for key in args_nni.keys()]), 'need only valid parameters'
        args_dict = vars(args)
        # cast params that should be int to int if needed (nni may offer them as float)
        args_nni_casted = {key:(int(value) if type(args_dict[key]) is int else value)
                            for key,value in args_nni.items()}
        args_dict.update(args_nni_casted)

        # adjust paths to NNI_OUTPUT_DIR (overrides passed args)
        nni_output_dir = os.path.expandvars('$NNI_OUTPUT_DIR')
        for param in ['save_model_path', 'prediction_output_path']:
            nni_path = os.path.join(nni_output_dir, os.path.basename(args_dict[param]))
            args_dict[param] = nni_path
        return args

def main():
    # ----------
    # args
    # ----------
    
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--model', type=str, choices=['dense'], default='dense',
                         help='''Model architecture. 
                                 dense - DenseNet, simple feed forward NN''')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--test-batch-size', type=int, default=512, help='Larger batch size for validation data')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    # data
    parser.add_argument('--data', type=str, help='should enlist train_data.h5, valid_data.h5, (test_data.h5)')
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
    parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false')
    parser.set_defaults(early_stopping=True)
    parser.add_argument('--patience', type=int, default=3, 
                         help='Epochs to wait before early stopping')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--nni', action='store_true')
    # store and load
    parser.add_argument('--save-model-path', type=str, default='./best_model.pt')
    parser.add_argument('--prediction-output-path', type=str, default='best_predictions.h5')
    parser.add_argument('--load-model-path', type=str, default='')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = HyperviewDataModule.add_dataloader_specific_args(parser)

    # add model specific args depending on chosen model
    temp_args, _ = parser.parse_known_args()
    if temp_args.model=='dense':
        parser = DenseNet.add_model_specific_args(parser)

    args = parser.parse_args()

    if args.nni:
        args = HyperviewMetricCallbacks.add_nni_params(args)

    if args.verbose:
        print('BEGIN argparse key - value pairs')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(args))
        print('END argparse key - value pairs')

    if args.load_model_path:
        print('INFERENCE MODE')
        print(f'loading model from {args.load_model_path}')

        # ----------
        # data
        # ----------

        # load arg Namespace from checkpoint
        print('command line arguments will be replaced with checkpoint["hyper_parameters"]')
        checkpoint = torch.load(args.load_model_path)
        checkpoint_args = argparse.Namespace(**checkpoint["hyper_parameters"])

        # potentially overwrite the data arg
        if args.data:
            checkpoint_args.data = args.data
            print(f'overwriting checkpoint argument: data dir = {checkpoint_args.data}')

        cdm = HyperviewDataModule(checkpoint_args)
        cdm.setup(stage='test')
        test_loader = cdm.test_dataloader()
        input_shapes = cdm.get_input_shapes(stage='test')
        
        if args.verbose:
            print('Input shapes', input_shapes)

        if checkpoint_args.model=='dense':
            backbone = DenseNet(checkpoint_args, input_shapes)
        # load model state from checkpoint
        model = HyperviewNet(backbone)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        trainer = pl.Trainer(weights_summary='full', 
                             num_sanity_val_steps=0, 
                             enable_progress_bar=False)
        trainer.test(model=model, test_dataloaders=test_loader)
        y_pred = trainer.predict(model=model, dataloaders=[test_loader])
        y_pred = torch.cat(y_pred).detach().cpu().numpy().squeeze()
        
    else:
        print('TRAINING MODE')

        # ----------
        # data
        # ----------

        cdm = HyperviewDataModule(args)
        cdm.setup(stage='fit')
        train_loader = cdm.train_dataloader()
        valid_loader = cdm.val_dataloader()

        input_shapes = cdm.get_input_shapes() 
    
        if args.verbose:
            print('Input shapes', input_shapes)
        # ----------
        # model
        # ----------
        if args.model=='dense':
            model = HyperviewNet(DenseNet(args, input_shapes))

        # ----------
        # training
        # ----------
        callbacks = [HyperviewMetricCallbacks(args), ModelSummary(max_depth=-1)] # model checkpoint is a model callback
        if args.early_stopping:
            callbacks.append(EarlyStopping(monitor='valid_loss', patience=args.patience, mode='min'))

        trainer = pl.Trainer.from_argparse_args(args, 
                                                fast_dev_run=False, # debug option
                                                logger=False,
                                                callbacks=callbacks, 
                                                enable_progress_bar=False,
                                                num_sanity_val_steps=0) # skip validation check
        trainer.fit(model, train_loader, valid_loader)

        best_epoch = int(trainer.callback_metrics["best_epoch"])
        ckpt_path = HyperviewNet.best_checkpoint_path(args.save_model_path, best_epoch)
        print(f'\nLoading best model from {ckpt_path}')
        trainer.validate(dataloaders=valid_loader, ckpt_path=ckpt_path)
        #model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        model.eval()
        # make predictions on *validation set*
        y_pred = trainer.predict(model=model, dataloaders=[valid_loader])
        y_pred = torch.cat(y_pred).detach().cpu().numpy().squeeze()
        # save model -- TODO breaks
        #script = model.to_torchscript()
        #torch.jit.save(script, args.save_model_path)


    # procedures that take place in fit and in test stage
    # save predictions
    # TODO save predictions y_pred in the required json format

if __name__=='__main__':
    main()
