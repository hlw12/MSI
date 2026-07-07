import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# layer_cols index mapping for the 12 feature columns:
# acc:  L1[2,3] L2[6,7]  L3[10,11]  -> each layer: [acc_ns, acc_ew]
# disp: L1[0,1] L2[4,5]  L3[8,9]    -> each layer: [disp_ns, disp_ew]
# layer_combo: [0,1,2] maps to Layer1/Layer2/Layer3 and can be customized.

_ACC_LAYER_BASE  = {0: [2, 3],  1: [6, 7],  2: [10, 11]}
_DISP_LAYER_BASE = {0: [0, 1],  1: [4, 5],  2: [8,  9]}


def _resolve_layer_cols(data_type: str, layer_combo: list) -> list:
    base = _ACC_LAYER_BASE if data_type == 'acc' else _DISP_LAYER_BASE
    cols = []
    for l in sorted(layer_combo):
        cols.extend(base[l])
    return cols


class SeismicDataset(Dataset):
    def __init__(self,
                 root_dir,
                 csv_path,
                 data_type='disp',
                 layer_combo=None,
                 window_size=1024,
                 stride=512,
                 split='train',
                 normalize=True,
                 external_scale=None):

        assert data_type in ['acc', 'disp']
        if layer_combo is None:
            layer_combo = [0, 1, 2]
        assert all(l in [0, 1, 2] for l in layer_combo) and len(layer_combo) >= 1

        self.root_dir    = root_dir
        self.data_type   = data_type
        self.layer_combo = layer_combo
        self.window_size = window_size
        self.stride      = stride
        self.normalize   = normalize
        self.split       = split
        self.scale       = 1.0
        self.samples     = []

        self.ground_cols = [0, 1] if data_type == 'acc' else [2, 3]
        self.layer_cols  = _resolve_layer_cols(data_type, layer_combo)
        self.filter_col  = 'A' if data_type == 'acc' else 'D'
        self.condition_channels = len(self.layer_cols)   # Used for model initialization.

        self.target_files = self._parse_csv(csv_path, split)
        self._load_and_slice()

        if self.normalize:
            if external_scale is not None:
                self.scale = external_scale
            else:
                self._compute_stats()
            self._normalize_samples()

    def _parse_csv(self, csv_path, split):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        valid_files = [
            {'filename': str(row['filename']).strip(),
             'folder':   str(row['source_folder']).strip()}
            for _, row in df.iterrows()
            if row[self.filter_col] == 1
        ]
        cut = int(len(valid_files) * 0.8)
        return valid_files[:cut] if split == 'train' else valid_files[cut:]

    def _load_and_slice(self):
        for meta in self.target_files:
            fname = meta['filename']
            if not fname.endswith('.npz'):
                fname += '.npz'
            fpath = os.path.join(self.root_dir, meta['folder'], fname)
            if not os.path.exists(fpath):
                continue
            try:
                data       = np.load(fpath, allow_pickle=True)
                ground_seq = data['labels'][:, self.ground_cols]
                layer_seq  = data['features'][:, self.layer_cols]
                total_len  = ground_seq.shape[0]
                for start in range(0, total_len - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    self.samples.append({
                        'ground': ground_seq[start:end].copy(),
                        'layers': layer_seq[start:end].copy(),
                    })
            except Exception as e:
                print(f"Error loading {fname}: {e}")

    def _compute_stats(self):
        if not self.samples:
            return
        all_g = np.concatenate([s['ground'] for s in self.samples], axis=0)
        all_l = np.concatenate([s['layers'] for s in self.samples], axis=0)
        self.scale = float(max(np.abs(all_g).max(), np.abs(all_l).max())) + 1e-6

    def _normalize_samples(self):
        for s in self.samples:
            s['ground'] = (s['ground'] / self.scale).astype(np.float32)
            s['layers'] = (s['layers'] / self.scale).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'ground': torch.from_numpy(s['ground']).float().T,
            'layers': torch.from_numpy(s['layers']).float().T,
            'scale':  self.scale,
        }