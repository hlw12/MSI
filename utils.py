import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def save_plot(real, inverted, epoch, save_path):
    time = np.arange(real.shape[-1])
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].set_title(f"Epoch {epoch} - NS")
    ax[0].plot(time, real[0],     color='black', alpha=0.6, label='GT')
    ax[0].plot(time, inverted[0], color='red',   linestyle='--', label='Pred')
    ax[0].legend()
    ax[1].set_title(f"Epoch {epoch} - EW")
    ax[1].plot(time, real[1],     color='black', alpha=0.6, label='GT')
    ax[1].plot(time, inverted[1], color='blue',  linestyle='--', label='Pred')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
