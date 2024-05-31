import torch
from torch import nn
from opt import get_opts

# datasets
from dataset import AudioDataset
from torch.utils.data import DataLoader

# models
from models import PE, MLP, KAN, Siren, GaborNet, MultiscaleBACON

# metrics
from metrics import mse

# optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(42, workers=True)

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class CoordMLPSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.validation_step_outputs = []

        if hparams.use_pe:
            P = torch.cat([torch.tensor(1)*2**i for i in range(10)], 1) # (2, 2*10)
            self.pe = PE(P)

        if hparams.arch in ['relu', 'gaussian', 'quadratic',
                            'multi-quadratic', 'laplacian',
                            'super-gaussian', 'expsin']:
            kwargs = {'a': hparams.a, 'b': hparams.b}
            act = hparams.arch
            if hparams.use_pe:
                n_in = self.pe.out_dim
            else:
                n_in = 1
            self.mlp = MLP(n_in=n_in, act=act, n_out=hparams.n_out,
                           act_trainable=hparams.act_trainable,
                           **kwargs)

        elif hparams.arch == 'ff':
            P = hparams.sc*torch.normal(torch.zeros(1, 256),
                                        torch.ones(1, 256)) # (2, 256)
            self.pe = PE(P)
            self.mlp = MLP(n_in=self.pe.out_dim,
                           n_out=hparams.n_out)

        elif hparams.arch == 'siren':
            self.mlp = Siren(in_features=1,
                             first_omega_0=hparams.omega_0,
                             hidden_omega_0=hparams.omega_0,
                             out_features=hparams.n_out)

        elif hparams.arch == 'gabor':
            self.mlp = GaborNet(in_size=1,
                                input_scale=max(hparams.img_wh)/4, 
                                out_size=hparams.n_out)

        elif hparams.arch == 'bacon':
            self.mlp = MultiscaleBACON(
                    frequency=[hparams.img_wh[0]//4, hparams.img_wh[1]//4], out_size=hparams.n_out)
        
        elif hparams.arch == 'kan':
            if hparams.use_pe:
                n_in = self.pe.out_dim
            else:
                n_in = 1
            self.mlp = KAN(n_in=n_in,
                           n_out=hparams.n_out,
                           n_layers=hparams.kan_layers,
                           n_hidden_units=hparams.kan_hidden_dim,
                           basis=hparams.kan_basis,
                           use_kan_pe=hparams.use_kan_pe)

        print("Model: ", self.mlp)

    def forward(self, x):
        if hparams.use_pe or hparams.arch=='ff':
            x = self.pe(x)
        return self.mlp(x)
        
    def setup(self, stage=None):
        self.dataset = AudioDataset(wav_path=hparams.wav_path)

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        self.opt = Adam(self.mlp.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(self.opt, hparams.num_epochs, hparams.lr/1e2)

        return [self.opt], [scheduler]

    def training_step(self, batch, batch_idx):
        a_pred = self(batch['t'])['model_out']

        if hparams.arch=='bacon':
            loss = sum(mse(x, batch['a']) for x in a_pred)
        else:
            loss = mse(a_pred, batch['a'])

        psnr_ = -10*torch.log10(loss)

        self.log('lr', self.opt.param_groups[0]['lr'])
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        a_pred = self(batch['t'])['model_out']

        if hparams.arch=='bacon':
            loss = mse(a_pred[-1], batch['a'], reduction='none')
        else:
            loss = mse(a_pred, batch['a'], reduction='none')

        log = {'val_loss': loss,
               't': batch['t'],
               'a_gt': batch['a']}

        if hparams.arch=='bacon':
            log['a_pred'] = a_pred[-1]
        else:
            log['a_pred'] = a_pred

        self.validation_step_outputs.append(log)

    def on_validation_epoch_end(self):
        mean_loss = torch.cat([x['val_loss'] for x in self.validation_step_outputs]).mean()
        mean_psnr = -10*torch.log10(mean_loss)
        a_gt = torch.cat([x['a_gt'] for x in self.validation_step_outputs])
        a_pred = torch.cat([x['a_pred'] for x in self.validation_step_outputs])
        
        t = torch.cat([x['t'] for x in self.validation_step_outputs])
        

        fig, axes = plt.subplots(1,2)
        axes[0].plot(t.squeeze().detach().cpu().numpy(), a_gt.squeeze().detach().cpu().numpy())
        axes[1].plot(t.squeeze().detach().cpu().numpy(), a_pred.squeeze().detach().cpu().numpy())       
    
        self.logger.experiment.add_figure('val/gt_pred',
                                          fig,
                                          self.global_step)

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        
        self.validation_step_outputs.clear()  # free memory

if __name__ == '__main__':
    hparams = get_opts()
    system = CoordMLPSystem(hparams)

    pbar = TQDMProgressBar(refresh_rate=1)
    model_summary = ModelSummary(max_depth=-1)
    callbacks = [pbar, model_summary]

    logger = TensorBoardLogger(save_dir=hparams.save_dir,
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=0,
                      log_every_n_steps=1,
                      check_val_every_n_epoch=20,
                      benchmark=True)

    trainer.fit(system)