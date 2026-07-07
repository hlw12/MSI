import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from net.component import SinusoidalPositionEmbeddings, Block1D


class ConditionalDiffusionUNet(nn.Module):
    def __init__(self, ground_channels=2, condition_channels=6, time_emb_dim=128):
        super().__init__()
        in_ch = ground_channels + condition_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.inc  = Block1D(in_ch, 64,  time_emb_dim)
        self.down1 = Block1D(64,  128, time_emb_dim)
        self.down2 = Block1D(128, 256, time_emb_dim)
        self.bot   = Block1D(256, 512, time_emb_dim)

        self.up1  = nn.ConvTranspose1d(512, 256, 2, stride=2)
        self.dec1 = Block1D(512, 256, time_emb_dim)
        self.up2  = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec2 = Block1D(256, 128, time_emb_dim)
        self.up3  = nn.ConvTranspose1d(128, 64,  2, stride=2)
        self.dec3 = Block1D(128, 64,  time_emb_dim)

        self.outc = nn.Conv1d(64, ground_channels, 1)

    def forward(self, x, condition, t):
        t_emb   = self.time_mlp(t)
        x_input = torch.cat([x, condition], dim=1)

        x1 = self.inc(x_input, t_emb)
        x2 = self.down1(F.max_pool1d(x1, 2), t_emb)
        x3 = self.down2(F.max_pool1d(x2, 2), t_emb)
        x4 = self.bot(F.max_pool1d(x3, 2), t_emb)

        def _up(feat, skip, up_layer, dec_layer):
            feat = up_layer(feat)
            if feat.size(2) != skip.size(2):
                feat = F.interpolate(feat, size=skip.size(2))
            return dec_layer(torch.cat([skip, feat], dim=1), t_emb)

        x = _up(x4, x3, self.up1, self.dec1)
        x = _up(x,  x2, self.up2, self.dec2)
        x = _up(x,  x1, self.up3, self.dec3)
        return self.outc(x)


class SubVPDiffusionManager:
    def __init__(self, timesteps=1000, device='cuda'):
        self.timesteps = timesteps
        self.device    = device
        self.beta      = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alpha     = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.mean_coef = torch.sqrt(self.alpha_hat)
        self.std       = 1.0 - self.alpha_hat

    def add_noise(self, x0, t):
        mean    = self.mean_coef[t][:, None, None]
        std     = self.std[t][:, None, None]
        epsilon = torch.randn_like(x0)
        return mean * x0 + std * epsilon, epsilon


def freq_weighted_loss(pred_noise, noise, freq_alpha=0.5, fs=100.0, f_cut=10.0):
    """
    Frequency-weighted loss: time-domain MSE plus frequency-domain weighted MSE.

    Earthquake engineering focuses on the low-frequency band around 0.1-10 Hz.
    Standard MSE treats all frequencies equally. This function gives higher
    weights to low-frequency components in the frequency domain, encouraging the
    model to prioritize the frequency range of engineering interest and improve
    the eSa metric.

    Parameters
    ----------
    pred_noise : [B, 2, T]
        Predicted noise from the model.
    noise : [B, 2, T]
        Ground-truth noise.
    freq_alpha : float
        Weight of the time-domain loss. The frequency-domain loss weight is
        1 - freq_alpha. Defaults to 0.5.
    fs : float
        Sampling frequency in Hz, used to construct the frequency axis.
    f_cut : float
        Cutoff frequency in Hz. Components below this frequency receive higher
        weights.

    Returns
    -------
    scalar loss
    """
    loss_time = F.mse_loss(pred_noise, noise)

    T       = pred_noise.shape[-1]
    freqs   = torch.fft.rfftfreq(T, d=1.0/fs).to(pred_noise.device)  # [F]
    weight  = torch.where(freqs < f_cut,
                          torch.ones_like(freqs),
                          torch.full_like(freqs, 0.1))                # [F]

    pred_fft = torch.fft.rfft(pred_noise, dim=-1)   # [B, 2, F] complex
    noise_fft = torch.fft.rfft(noise,     dim=-1)   # [B, 2, F] complex
    diff_abs  = (pred_fft - noise_fft).abs()        # [B, 2, F]
    loss_freq = (weight * diff_abs).mean()

    return freq_alpha * loss_time + (1.0 - freq_alpha) * loss_freq


class DDIMSampler:
    def __init__(self, diffusion_manager: SubVPDiffusionManager, ddim_steps=50):
        self.dm   = diffusion_manager
        total     = self.dm.timesteps
        step      = total // ddim_steps
        self.timestep_seq = list(reversed(range(0, total, step)))[:ddim_steps]

    def _predict_x0(self, x_t, t_idx, pred_noise):
        mean  = self.dm.mean_coef[t_idx]
        std   = self.dm.std[t_idx]
        x0    = (x_t - std * pred_noise) / (mean + 1e-8)
        clamp = min(3.0 / (mean.item() + 0.1), 5.0)
        return torch.clamp(x0, -clamp, clamp)

    def _ddim_step(self, x_t, t_now, t_prev, pred_noise, eta=0.0):
        x0_pred   = self._predict_x0(x_t, t_now, pred_noise)
        mean_prev = self.dm.mean_coef[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        std_prev  = self.dm.std[t_prev]       if t_prev >= 0 else torch.tensor(0.0, device=x_t.device)
        x_prev    = mean_prev * x0_pred + std_prev * pred_noise
        if eta > 0.0 and t_prev >= 0:
            x_prev = x_prev + eta * std_prev * torch.randn_like(x_prev)
        return x_prev

    @torch.no_grad()
    def sample(self, model, condition, device, eta=0.0, verbose=False):
        B   = condition.shape[0]
        x   = torch.randn(B, 2, 1024, device=device)
        seq = self.timestep_seq
        itr = tqdm(range(len(seq)), desc="DDIM Sampling") if verbose else range(len(seq))
        for i in itr:
            t_now  = seq[i]
            t_prev = seq[i + 1] if (i + 1) < len(seq) else -1
            t_t    = torch.full((B,), t_now, device=device, dtype=torch.long)
            x      = self._ddim_step(x, t_now, t_prev, model(x, condition, t_t), eta=eta)
        return x

    @torch.no_grad()
    def sample_ensemble(self, model, condition, device, n_samples=10, eta=1.0):
        """
        Ensemble sampling for uncertainty quantification.

        The method performs multiple stochastic sampling runs and returns the
        mean prediction, pointwise standard deviation, and all samples.

        Notes
        -----
        - eta=1.0 keeps stochasticity, so each sampling run can produce a
          different result.
        - The mean is used as the final prediction.
        - The standard deviation is used as pointwise uncertainty.
        - The 95% confidence interval is approximately mean +/- 1.96 * std.

        Parameters
        ----------
        n_samples : int
            Number of sampling runs. A typical choice is 10-20.
        eta : float
            Stochasticity coefficient. Use 1.0 for uncertainty quantification.

        Returns
        -------
        mean : [B, 2, 1024]
            Mean prediction.
        std : [B, 2, 1024]
            Pointwise standard deviation.
        all : [n, B, 2, 1024]
            All samples.
        """
        preds = [self.sample(model, condition, device, eta=eta) for _ in range(n_samples)]
        stack = torch.stack(preds, dim=0)   # [n, B, 2, 1024]
        return stack.mean(0), stack.std(0), stack
