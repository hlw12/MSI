import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import SeismicDataset
from net.Diffusion import ConditionalDiffusionUNet, SubVPDiffusionManager, DDIMSampler


def plot_results(real, inverted):
    """real and inverted: [2, 1024], restored to physical units."""
    time = np.arange(real.shape[-1])
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].set_title("Inversion Result - NS Direction")
    ax[0].plot(time, real[0],     label='Ground Truth', color='black', alpha=0.6, linewidth=1.5)
    ax[0].plot(time, inverted[0], label='Inverted (AI)', color='red',  linestyle='--', linewidth=1.5)
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].set_title("Inversion Result - EW Direction")
    ax[1].plot(time, real[1],     label='Ground Truth', color='black', alpha=0.6, linewidth=1.5)
    ax[1].plot(time, inverted[1], label='Inverted (AI)', color='blue', linestyle='--', linewidth=1.5)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("inversion_result.png", dpi=150)
    print("Result saved to inversion_result.png")


def invert_ground_motion(
    val_dataset,
    model_path,
    device='cuda',
    ddim_steps=50,
    num_ensemble=1,
    eta=0.0
):
    idx       = 0
    sample    = val_dataset[idx]
    condition = sample['layers'].unsqueeze(0).to(device)
    gt_ground = sample['ground'].unsqueeze(0)
    scale     = sample['scale']

    model = ConditionalDiffusionUNet(ground_channels=2, condition_channels=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from: {model_path}")
    print(f"Scale factor: {scale:.4f}  |  DDIM steps: {ddim_steps}")

    dm      = SubVPDiffusionManager(timesteps=1000, device=device)
    sampler = DDIMSampler(dm, ddim_steps=ddim_steps)

    results = []
    for run in range(num_ensemble):
        x = sampler.sample(model, condition, device, eta=eta, verbose=(run == 0))
        results.append(x)

    inverted    = torch.stack(results, dim=0).mean(dim=0)
    inverted_np = inverted.cpu().numpy()[0] * scale
    gt_np       = gt_ground.numpy()[0] * scale

    plot_results(gt_np, inverted_np)

    rmse    = np.sqrt(np.mean((inverted_np - gt_np) ** 2))
    corr_ns = np.corrcoef(gt_np[0], inverted_np[0])[0, 1]
    corr_ew = np.corrcoef(gt_np[1], inverted_np[1])[0, 1]
    print(f"\nEvaluation Metrics")
    print(f"RMSE:    {rmse:.6f}")
    print(f"Corr NS: {corr_ns:.4f}")
    print(f"Corr EW: {corr_ew:.4f}")
    print("-" * 30)

    return inverted_np, gt_np


if __name__ == '__main__':


    DATA_ROOT  = "<DATASET_ROOT>"
    # Replace this placeholder with your local dataset root directory.
    CSV_PATH   = DATA_ROOT + "/<DATASET_ANNOTATION_CSV>"
    # Replace this placeholder with your dataset annotation CSV file name.
    MODEL_PATH = r"experiments/run_001/checkpoints/best_model.pth"
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    val_ds = SeismicDataset(DATA_ROOT, CSV_PATH, data_type='disp', split='val')

    invert_ground_motion(
        val_dataset=val_ds,
        model_path=MODEL_PATH,
        device=DEVICE,
        ddim_steps=50,
        num_ensemble=1,
        eta=0.0
    )