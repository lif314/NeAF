import torch
from opt import get_opts

# datasets
from dataset import AudioDataset
from torch.utils.data import DataLoader

# models
from models.mlp import MLP
from models.siren import Siren
from models.incode import INCODE
from models.wire import Wire
from models.fourier_kan import FourierKAN
from models.bspline_kan import BsplineKAN
from models.hyper_kan import HyperKAN

# metrics
from metrics import mse, calc_snr, compute_log_distortion

# optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

import librosa
import soundfile as sf
import os

from encoding import Encoding

seed_everything(42, workers=True)

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class CoordMLPSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.validation_step_outputs = []

        # Positional Encoding
        if hparams.pe_type == "None":
            # No Encoding
            pos_encode_configs = {'type': None}
        elif hparams.pe_type == "NeRF":
            # Frequency Encoding
            pos_encode_configs = {'type':'frequency', 'use_nyquist': True, 'mapping_input': hparams.batch_size}
        elif hparams.pe_type == "FFN":
            # Gaussian Encoding
            pos_encode_configs = {'type':'gaussian', 'scale_B': 100, 'mapping_input': 32}

        self.pos_encode = pos_encode_configs['type']
        if self.pos_encode in Encoding().encoding_dict.keys():
            self.positional_encoding = Encoding(self.pos_encode).run(in_features=hparams.in_features, pos_encode_configs=pos_encode_configs)
        elif self.pos_encode == None:
            self.pos_encode = False
        else:
            assert "Invalid pos_encode. Choose from: [frequency, gaussian]"

        if  self.pos_encode:
            print("PE Dim: ", self.positional_encoding.out_dim)

        if hparams.arch in ['relu', 'prelu', 'selu', 'tanh',
                            'sigmoid', 'silu', 'softplus', 'elu',
                            'sinc', 'gaussian', 'quadratic', 
                            'multi-quadratic', 'laplacian', 'super-gaussian', 'expsin']:
            kwargs = {'a': hparams.a, 'b': hparams.b}
            act = hparams.arch
            if self.pos_encode:
                n_in = self.positional_encoding.out_dim
            else:
                n_in = hparams.in_features
            self.net = MLP(in_features=n_in,
                           hidden_layers=4,
                           hidden_features=256,
                           act=act,
                           out_features=hparams.out_features,
                           act_trainable=hparams.act_trainable,
                           **kwargs)

        elif hparams.arch == 'siren':
            self.net = Siren(in_features=hparams.in_features,
                             hidden_layers=4,
                             hidden_features=256,
                             first_omega_0=hparams.first_omega_0,
                             hidden_omega_0=hparams.hidden_omega_0,
                             out_features=hparams.out_features)

        elif hparams.arch == 'incode':
            self.net = INCODE(in_features=hparams.in_features,
                             hidden_layers=4,
                             hidden_features=256,
                             first_omega_0=hparams.first_omega_0,
                             hidden_omega_0=hparams.hidden_omega_0,
                             out_features=hparams.out_features,
                             outermost_linear=True)
            
        elif hparams.arch == 'wire':
            self.net = Wire(in_features=hparams.in_features,
                             hidden_layers=4,
                             hidden_features=256,
                             wire_type='complex',
                             first_omega_0=hparams.first_omega_0,
                             hidden_omega_0=hparams.hidden_omega_0,
                             out_features=hparams.out_features,
                             sigma=10.0)

        elif hparams.arch == 'fourier':
            if self.pos_encode:
                n_in =  self.positional_encoding.out_dim
            else:
                n_in = hparams.in_features
            self.net = FourierKAN(in_features=n_in,
                                hidden_features=hparams.hidden_features,
                                hidden_layers=hparams.hidden_layers,
                                out_features=hparams.out_features,
                                input_grid_size=hparams.input_grid_size,
                                hidden_grid_size=hparams.hidden_grid_size,
                                output_grid_size=hparams.output_grid_size,
                                outermost_linear=hparams.outermost_linear,
                                init_type=hparams.init_type,
                                )
            
        elif hparams.arch == 'bspline':
            if self.pos_encode:
                n_in =  self.positional_encoding.out_dim
            else:
                n_in = hparams.in_features
            self.net = BsplineKAN(in_features=n_in,
                                hidden_features=hparams.hidden_features,
                                hidden_layers=hparams.hidden_layers,
                                out_features=hparams.out_features,
                                input_grid_size=hparams.input_grid_size,
                                hidden_grid_size=hparams.hidden_grid_size,
                                output_grid_size=hparams.output_grid_size,
                                )
            
        elif hparams.arch == 'hyper':
            if self.pos_encode:
                n_in =  self.positional_encoding.out_dim
            else:
                n_in = hparams.in_features
            self.net = HyperKAN(in_features=n_in,
                                hidden_features=hparams.hidden_features,
                                hidden_layers=hparams.hidden_layers,
                                out_features=hparams.out_features,
                                input_grid_size=hparams.input_grid_size,
                                hidden_grid_size=hparams.hidden_grid_size,
                                output_grid_size=hparams.output_grid_size,
                                outermost_linear=hparams.outermost_linear
                                )

        # print("Model: ", self.net)

    def forward(self, x):
        if self.pos_encode:
            x = self.positional_encoding(x)
        return self.net(x)
        
    def setup(self, stage=None):
        self.dataset = AudioDataset(dataset_name=hparams.dataset_name, audio_path=hparams.audio_path)
        self.rate = self.dataset.rate
        os.makedirs(os.path.join(self.logger.log_dir, "pred_wavs"), exist_ok=True)
        os.makedirs(os.path.join(self.logger.log_dir, "figs"), exist_ok=True)

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
        self.opt = Adam(self.net.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(self.opt, hparams.num_epochs, hparams.lr/1e2)

        return [self.opt], [scheduler]

    def training_step(self, batch, batch_idx):
        a_pred = self(batch['t'])

        loss = mse(a_pred, batch['a'])

        snr_ = calc_snr(a_pred, batch['a'])

        self.log('lr', self.opt.param_groups[0]['lr'])
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/snr', snr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        a_pred = self(batch['t'])

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
        a_gt = torch.cat([x['a_gt'] for x in self.validation_step_outputs])
        a_pred = torch.cat([x['a_pred'] for x in self.validation_step_outputs])

        mean_snr = calc_snr(a_pred.detach().clone(), a_gt)
        mean_lsd = compute_log_distortion(a_pred.detach().clone().cpu(), a_gt.cpu())
        
        t = torch.cat([x['t'] for x in self.validation_step_outputs])
        
        a_error = a_gt - a_pred

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].set_ylim(-1, 1)
        axes[1].set_ylim(-1, 1)
        error_threld = hparams.error_threld
        axes[2].set_ylim(-error_threld, error_threld)

        axes[0].yaxis.set_major_locator(FixedLocator([-1.0, -0.5,0.0, 0.5, 1.0]))
        axes[1].yaxis.set_major_locator(FixedLocator([-1.0, -0.5,0.0, 0.5, 1.0]))
        axes[2].yaxis.set_major_locator(FixedLocator([-error_threld ,0.0, error_threld]))

        axes[0].get_xaxis().set_visible(False)
        axes[1].get_xaxis().set_visible(False)
        axes[2].get_xaxis().set_visible(False)

        axes[0].plot(t.squeeze().detach().cpu().numpy(), a_gt.squeeze().detach().cpu().numpy())
        axes[1].plot(t.squeeze().detach().cpu().numpy(), a_pred.squeeze().detach().cpu().numpy())       
        axes[2].plot(t.squeeze().detach().cpu().numpy(), a_error.squeeze().detach().cpu().numpy())       
    
        self.logger.experiment.add_figure('val/gt_pred',
                                          fig,
                                          self.global_step)

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/snr', mean_snr, prog_bar=True)
        self.log('val/lsd', mean_lsd, prog_bar=True)

        if (self.current_epoch+1) % 200 == 0:
            audio_data = librosa.util.normalize(a_pred.squeeze().detach().cpu().numpy())
            filename = os.path.join(self.logger.log_dir, "pred_wavs", f"pred_epoch_{self.current_epoch}.wav")
            sf.write(filename, audio_data, self.rate)

            metrics_txt = []
            metrics_txt.append(f"epoch: {self.current_epoch}\n")
            metrics_txt.append(f"val/snr:  {mean_snr}\n")
            metrics_txt.append(f"val/lsd:  {mean_lsd}\n\n")

            with open(os.path.join(self.logger.log_dir, "metrics.txt"), "a") as file:
                file.writelines(metrics_txt)

            fig.savefig(os.path.join(self.logger.log_dir, "figs", f"{self.current_epoch}.png"))

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
                      check_val_every_n_epoch=hparams.check_val_every_n_epoch,
                      benchmark=True)

    trainer.fit(system)