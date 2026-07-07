import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import SeismicDataset

class DoubleConv(nn.Module):
    """(Conv1d => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ForwardModel1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=6):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        self.outc = nn.Conv1d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)  # [B, 32, 1024]
        x2 = self.down1(x1)  # [B, 64, 512]
        x3 = self.down2(x2)  # [B, 128, 256]
        x4 = self.down3(x3)  # [B, 256, 128]

        x = self.up1(x4)  # [B, 128, 256]
        if x.size(2) != x3.size(2):
            x = nn.functional.interpolate(x, size=x3.size(2), mode='linear', align_corners=True)

        x = torch.cat([x3, x], dim=1)  # [B, 128+128, 256]
        x = self.conv1(x)
        x = self.up2(x)  # [B, 64, 512]
        if x.size(2) != x2.size(2):
            x = nn.functional.interpolate(x, size=x2.size(2), mode='linear', align_corners=True)

        x = torch.cat([x2, x], dim=1)  # [B, 64+64, 512]
        x = self.conv2(x)
        x = self.up3(x)  # [B, 32, 1024]
        if x.size(2) != x1.size(2):
            x = nn.functional.interpolate(x, size=x1.size(2), mode='linear', align_corners=True)

        x = torch.cat([x1, x], dim=1)  # [B, 32+32, 1024]
        x = self.conv3(x)
        out = self.outc(x)  # [B, 6, 1024]
        return out

DATA_ROOT = "<DATASET_ROOT>"
# Replace this placeholder with your local dataset root directory.
CSV_PATH = os.path.join(DATA_ROOT, "<DATASET_ANNOTATION_CSV>")
# Replace this placeholder with your dataset annotation CSV file name.
SAVE_DIR = "checkpoints_forward"
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print(f"Using Device: {DEVICE}")
    train_ds = SeismicDataset(DATA_ROOT, CSV_PATH, data_type='acc', split='train')
    val_ds = SeismicDataset(DATA_ROOT, CSV_PATH, data_type='acc', split='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = ForwardModel1D(in_channels=2, out_channels=6).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    loss_history = {'train': [], 'val': []}
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            x = batch['ground'].to(DEVICE)
            y = batch['layers'].to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['ground'].to(DEVICE)
                y = batch['layers'].to(DEVICE)
                pred = model(x)
                val_loss += criterion(pred, y).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_forward_model.pth"))
            print(">>> Best Model Saved!")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history['train'], label='Train Loss')
    plt.plot(loss_history['val'], label='Val Loss')
    plt.title('Forward Model Training')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"))
    print("Training Complete.")


if __name__ == "__main__":
    train()

