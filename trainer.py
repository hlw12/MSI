import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List

from dataset import SeismicDataset
from net.Diffusion import (ConditionalDiffusionUNet, SubVPDiffusionManager,
                           DDIMSampler, freq_weighted_loss)
from reconstruct_full import reconstruct_all
from utils import save_plot, set_seed


@dataclass
class Config:
    # Paths
    DATA_ROOT: str  = "<DATASET_ROOT>"
    # Replace this placeholder with your local dataset root directory.
    SAVE_DIR:  str  = "experiments/run_001"

    # Data
    DATA_TYPE:   str       = 'disp'
    LAYER_COMBO: List[int] = None
    WINDOW_SIZE: int       = 1024
    STRIDE:      int       = 512

    BATCH_SIZE:  int   = 32
    LR:          float = 1e-4
    EPOCHS:      int   = 200
    NUM_WORKERS: int   = 4
    SEED:        int   = 42

    TIMESTEPS:   int   = 1000
    DDIM_STEPS:  int   = 50

    FREQ_ALPHA:  float = 0.5    
    FREQ_FS:     float = 100.0  
    FREQ_FCUT:   float = 10.0   

    UQ_SAMPLES:  int   = 10     
    UQ_ETA:      float = 1.0    

    VAL_INTERVAL: int  = 1
    VIS_INTERVAL: int  = 20
    FS:           float = 100.0

    def __post_init__(self):
        if self.LAYER_COMBO is None:
            self.LAYER_COMBO = [0, 1, 2]
        self.CSV_PATH = os.path.join(self.DATA_ROOT, "<DATASET_ANNOTATION_CSV>")
        # Replace this placeholder with your dataset annotation CSV file name.
        self.DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    def setup(self):
        for sub in ["", "visualizations", "checkpoints"]:
            os.makedirs(os.path.join(self.SAVE_DIR, sub), exist_ok=True)
        cfg_path = os.path.join(self.SAVE_DIR, "config.json")
        with open(cfg_path, 'w') as f:
            d = asdict(self)
            d['DEVICE'] = self.DEVICE
            json.dump(d, f, indent=2)
        print(f"Config saved → {cfg_path}")
        print(f"Device: {self.DEVICE} | DataType: {self.DATA_TYPE} | "
              f"Layers: {self.LAYER_COMBO} | freq_alpha: {self.FREQ_ALPHA}")


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.train_ds = SeismicDataset(
            cfg.DATA_ROOT, cfg.CSV_PATH,
            data_type=cfg.DATA_TYPE, layer_combo=cfg.LAYER_COMBO,
            window_size=cfg.WINDOW_SIZE, stride=cfg.STRIDE, split='train',
        )
        self.val_ds = SeismicDataset(
            cfg.DATA_ROOT, cfg.CSV_PATH,
            data_type=cfg.DATA_TYPE, layer_combo=cfg.LAYER_COMBO,
            window_size=cfg.WINDOW_SIZE, stride=cfg.STRIDE, split='val',
            external_scale=self.train_ds.scale,
        )
        self.train_loader = DataLoader(
            self.train_ds, batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg.BATCH_SIZE,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
        )

        self.dm      = SubVPDiffusionManager(timesteps=cfg.TIMESTEPS, device=cfg.DEVICE)
        self.sampler = DDIMSampler(self.dm, ddim_steps=cfg.DDIM_STEPS)
        self.model   = ConditionalDiffusionUNet(
            ground_channels=2,
            condition_channels=self.train_ds.condition_channels,
        ).to(cfg.DEVICE)

        self.optimizer     = optim.AdamW(self.model.parameters(), lr=cfg.LR)
        self.best_val_loss = float('inf')
        self.fixed_sample  = self.val_ds[0]

    # Training
    def train_one_epoch(self, epoch):
        self.model.train()
        total = 0.0
        pbar  = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.EPOCHS}")
        for batch in pbar:
            ground = batch['ground'].to(self.cfg.DEVICE)
            layers = batch['layers'].to(self.cfg.DEVICE)
            t      = torch.randint(1, self.cfg.TIMESTEPS, (ground.shape[0],),
                                   device=self.cfg.DEVICE)
            noisy, noise = self.dm.add_noise(ground, t)
            pred_noise   = self.model(noisy, layers, t)

            loss = freq_weighted_loss(
                pred_noise, noise,
                freq_alpha=self.cfg.FREQ_ALPHA,
                fs=self.cfg.FREQ_FS,
                f_cut=self.cfg.FREQ_FCUT,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        return total / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total = 0.0
        for batch in self.val_loader:
            ground = batch['ground'].to(self.cfg.DEVICE)
            layers = batch['layers'].to(self.cfg.DEVICE)
            t      = torch.randint(1, self.cfg.TIMESTEPS, (ground.shape[0],),
                                   device=self.cfg.DEVICE)
            noisy, noise = self.dm.add_noise(ground, t)
            pred_noise   = self.model(noisy, layers, t)
            total += freq_weighted_loss(
                pred_noise, noise,
                freq_alpha=self.cfg.FREQ_ALPHA,
                fs=self.cfg.FREQ_FS,
                f_cut=self.cfg.FREQ_FCUT,
            ).item()
        return total / len(self.val_loader)

    @torch.no_grad()
    def sample_visualization(self, epoch):
        self.model.eval()
        condition = self.fixed_sample['layers'].unsqueeze(0).to(self.cfg.DEVICE)
        scale     = self.fixed_sample['scale']
        x         = self.sampler.sample(self.model, condition, self.cfg.DEVICE)
        save_plot(
            self.fixed_sample['ground'].numpy() * scale,
            x.cpu().numpy()[0] * scale,
            epoch,
            os.path.join(self.cfg.SAVE_DIR, "visualizations", f"epoch_{epoch}.png"),
        )

    @torch.no_grad()
    def uq_visualization(self):
        self.model.eval()
        condition = self.fixed_sample['layers'].unsqueeze(0).to(self.cfg.DEVICE)
        scale     = self.fixed_sample['scale']
        gt        = self.fixed_sample['ground'].numpy() * scale   # [2, 1024]

        mean, std, _ = self.sampler.sample_ensemble(
            self.model, condition, self.cfg.DEVICE,
            n_samples=self.cfg.UQ_SAMPLES, eta=self.cfg.UQ_ETA,
        )
        mean_np = mean.cpu().numpy()[0] * scale   # [2, 1024]
        std_np  = std.cpu().numpy()[0]  * scale   # [2, 1024]
        time    = np.arange(1024)

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        for ch, (ax, d) in enumerate(zip(axs, ['NS', 'EW'])):
            ax.plot(time, gt[ch],       color='black', lw=1.2, label='Ground Truth')
            ax.plot(time, mean_np[ch],  color='#D95319', lw=1.0, ls='--', label='Pred Mean')
            ax.fill_between(time,
                            mean_np[ch] - 1.96 * std_np[ch],
                            mean_np[ch] + 1.96 * std_np[ch],
                            color='#D95319', alpha=0.2, label='95% CI')
            ax.set_title(f'{d} Direction  (N={self.cfg.UQ_SAMPLES} samples)')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.suptitle("Ensemble UQ — Uncertainty Quantification", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.SAVE_DIR, "uq_visualization.png"), dpi=150)
        plt.close()
        print(f"UQ visualization saved → {self.cfg.SAVE_DIR}/uq_visualization.png")

    # Main loop
    def run(self):
        train_losses, val_losses = [], []

        for epoch in range(1, self.cfg.EPOCHS + 1):
            train_losses.append(self.train_one_epoch(epoch))

            if epoch % self.cfg.VAL_INTERVAL == 0:
                val_loss = self.validate()
                val_losses.append(val_loss)
                print(f"Epoch {epoch}  val_loss={val_loss:.6f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(),
                               os.path.join(self.cfg.SAVE_DIR, "checkpoints", "best_model.pth"))

            if epoch % self.cfg.VIS_INTERVAL == 0:
                self.sample_visualization(epoch)

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(self.cfg.SAVE_DIR, "checkpoints", f"epoch_{epoch}.pth"))

        best_path = os.path.join(self.cfg.SAVE_DIR, "checkpoints", "best_model.pth")
        self.model.load_state_dict(
            torch.load(best_path, map_location=self.cfg.DEVICE, weights_only=True)
        )

        self.uq_visualization()
        reconstruct_all(
            model=self.model, ddim_sampler=self.sampler,
            val_ds=self.val_ds, device=self.cfg.DEVICE,
            save_dir=self.cfg.SAVE_DIR, fs=self.cfg.FS,
            window_size=self.cfg.WINDOW_SIZE, stride=self.cfg.STRIDE,
            data_type=self.cfg.DATA_TYPE,   
        )
        plt.figure()
        plt.plot(train_losses, label='Train')
        plt.plot(
            [i * self.cfg.VAL_INTERVAL for i in range(1, len(val_losses)+1)],
            val_losses, label='Val'
        )
        plt.legend(); plt.title("Loss Curve")
        plt.savefig(os.path.join(self.cfg.SAVE_DIR, "loss_curve.png"))
        plt.close()
        print("Training finished.")


if __name__ == '__main__':
    cfg = Config(
        DATA_TYPE    = 'acc',
        LAYER_COMBO  = [0,2],
        SAVE_DIR     = 'experiments/run_acc02_2',
        FREQ_ALPHA   = 0.5,   
        FREQ_FCUT    = 10.0,   # Frequencies below 10 Hz are treated as low-frequency components.
        UQ_SAMPLES   = 10,
    )
    cfg.setup()
    set_seed(cfg.SEED)
    Trainer(cfg).run()
