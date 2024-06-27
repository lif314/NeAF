import torch
from opt import get_opts

# datasets
from dataset import AudioDataset
from torch.utils.data import DataLoader

# models
from models import PE, MLP, FourierKAN, Siren

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
            P = [torch.tensor(1)*2**i for i in range(10)] # (1, 2*10)
            self.pe = PE(P)

        if hparams.arch in ['relu', 'tanh', 'sinc', 'softsign', 'gaussian', 'quadratic',
                            'multi-quadratic', 'laplacian',
                            'super-gaussian', 'expsin']:
            kwargs = {'a': hparams.a, 'b': hparams.b}
            act = hparams.arch
            if hparams.use_pe:
                n_in = self.pe.out_dim
            else:
                n_in = hparams.in_features
            self.mlp = MLP(n_in=n_in,
                           act=act,
                           n_out=hparams.out_features,
                           act_trainable=hparams.act_trainable,
                           **kwargs)

        elif hparams.arch == 'ff':
            P = hparams.sc * torch.normal(torch.zeros(1, 256))  # (1, 256)
            self.pe = PE(P)
            self.mlp = MLP(n_in=self.pe.out_dim,
                           n_out=hparams.out_features)

        elif hparams.arch == 'siren': # 263K
            self.mlp = Siren(in_features=hparams.in_features,
                            #  hidden_layers=hparams.hidden_layers,
                             hidden_layers=4,
                            #  hidden_features=hparams.hidden_features,
                             hidden_features=256,
                             first_omega_0=hparams.first_omega_0,
                             hidden_omega_0=hparams.hidden_omega_0,
                             out_features=hparams.out_features)
        
        elif hparams.arch == 'fourier':
            if hparams.use_pe:
                n_in = self.pe.out_dim
            else:
                n_in = hparams.in_features
            self.mlp = FourierKAN( in_features=n_in,
                        hidden_features=hparams.hidden_features,
                        hidden_layers=hparams.hidden_layers,
                        out_features=hparams.out_features,
                        input_grid_size=hparams.input_grid_size,
                        hidden_grid_size=hparams.hidden_grid_size,
                        output_grid_size=hparams.output_grid_size
                    )

        print("Model: ", self.mlp)

    def forward(self, x):
        if hparams.use_pe or hparams.arch=='ff':
            x = self.pe(x)
        return self.mlp(x)
        
    def setup(self, stage=None):
        self.dataset = AudioDataset(dataset_name=hparams.dataset_name, audio_path=hparams.audio_path)
        self.rate = self.dataset.rate
        os.makedirs(os.path.join(self.logger.log_dir, "pred_wavs"), exist_ok=True)
        os.makedirs(os.path.join(self.logger.log_dir, "figs"), exist_ok=True)

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          shuffle=True,
                          num_workers=0,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        self.opt = Adam(self.mlp.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(self.opt, hparams.num_epochs, hparams.lr/1e2)

        return [self.opt], [scheduler]

    def training_step(self, batch, batch_idx):
        a_pred = self(batch['t'])['model_out']

        loss = mse(a_pred, batch['a'])

        snr_ = calc_snr(a_pred, batch['a'])

        self.log('lr', self.opt.param_groups[0]['lr'])
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/snr', snr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        a_pred = self(batch['t'])['model_out']

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
        # mean_psnr = -10*torch.log10(mean_loss)
        a_gt = torch.cat([x['a_gt'] for x in self.validation_step_outputs])
        a_pred = torch.cat([x['a_pred'] for x in self.validation_step_outputs])
        
        mean_snr = calc_snr(a_pred.detach().clone(), a_gt)
        mean_lsd = compute_log_distortion(a_pred.detach().clone().cpu(), a_gt.cpu())
        
        t = torch.cat([x['t'] for x in self.validation_step_outputs])
        
        a_error = a_gt - a_pred

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].set_ylim(-1, 1)
        axes[1].set_ylim(-1, 1)
        axes[2].set_ylim(-0.1, 0.1)

        axes[0].yaxis.set_major_locator(FixedLocator([-1.0, -0.5,0.0, 0.5, 1.0]))
        axes[1].yaxis.set_major_locator(FixedLocator([-1.0, -0.5,0.0, 0.5, 1.0]))
        axes[2].yaxis.set_major_locator(FixedLocator([-0.1 ,0.0, 0.1]))

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
                      check_val_every_n_epoch=20,
                      benchmark=True)

    trainer.fit(system)