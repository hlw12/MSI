import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt


def hampel_filter(sig, window=10, n_sigma=3.0):
    out = sig.copy()
    for i in range(len(out)):
        lo  = max(0, i - window)
        hi  = min(len(out), i + window + 1)
        med = np.median(out[lo:hi])
        mad = np.median(np.abs(out[lo:hi] - med))
        if mad < 1e-12:
            continue
        if np.abs(out[i] - med) > n_sigma * 1.4826 * mad:
            out[i] = med
    return out


def remove_spikes(sig2d, window=10, n_sigma=3.0):
    return np.stack([
        hampel_filter(sig2d[ch], window, n_sigma)
        for ch in range(sig2d.shape[0])
    ])


def hann_overlap_add(windows, window_size, stride, total_len):
    out    = np.zeros((2, total_len), dtype=np.float64)
    weight = np.zeros(total_len, dtype=np.float64)
    hann   = np.hanning(window_size)
    for m, w in enumerate(windows):
        start = m * stride
        end   = start + window_size
        if end > total_len:
            break
        out[:, start:end] += w * hann[np.newaxis, :]
        weight[start:end] += hann
    weight = np.where(weight < 1e-8, 1.0, weight)
    return out / weight[np.newaxis, :]


def compute_response_spectrum(acc, fs=100.0, damping=0.05, periods=None):
    """Compute the pseudo-acceleration spectrum PSa(T) using the Newmark-beta average acceleration method. Used only in acc mode."""
    if periods is None:
        periods = np.concatenate([
            np.linspace(0.01, 0.1,  19, endpoint=False),
            np.linspace(0.1,  1.0,  37, endpoint=False),
            np.linspace(1.0,  4.0,  31),
        ])
    dt       = 1.0 / fs
    Sa       = np.zeros(len(periods))
    for i, T in enumerate(periods):
        omega = 2.0 * np.pi / T
        if T < dt:
            Sa[i] = np.max(np.abs(acc))
            continue
        xi       = damping
        beta_nm  = 0.25
        gamma_nm = 0.5
        n        = len(acc)
        u = np.zeros(n)
        v = np.zeros(n)
        a = np.zeros(n)
        k_eff = (omega ** 2
                 + gamma_nm / (beta_nm * dt) * 2.0 * xi * omega
                 + 1.0 / (beta_nm * dt ** 2))
        for j in range(1, n):
            dp_eff = (
                -acc[j]
                + 1.0 / (beta_nm * dt ** 2) * u[j - 1]
                + 1.0 / (beta_nm * dt)       * v[j - 1]
                + (1.0 / (2.0 * beta_nm) - 1.0) * a[j - 1]
                + 2.0 * xi * omega * (
                    gamma_nm / (beta_nm * dt)                  * u[j - 1]
                    + (gamma_nm / beta_nm - 1.0)               * v[j - 1]
                    + dt * (gamma_nm / (2.0 * beta_nm) - 1.0) * a[j - 1]
                )
            )
            u[j] = dp_eff / k_eff
            a[j] = ((u[j] - u[j - 1]) / (beta_nm * dt ** 2)
                    - v[j - 1] / (beta_nm * dt)
                    - (1.0 / (2.0 * beta_nm) - 1.0) * a[j - 1])
            v[j] = (v[j - 1]
                    + dt * (1.0 - gamma_nm) * a[j - 1]
                    + dt * gamma_nm         * a[j])
        Sa[i] = omega ** 2 * np.max(np.abs(u))
    return periods, Sa


def invert_record(model, ddim_sampler, record_samples, device,
                  window_size=1024, stride=512, scale=1.0):
    pred_windows, gt_windows = [], []
    model.eval()
    with torch.no_grad():
        for sample in record_samples:
            layers_raw = sample['layers']
            ground_raw = sample['ground']
            if not isinstance(layers_raw, torch.Tensor):
                layers_raw = torch.from_numpy(layers_raw.astype(np.float32)).T
            if not isinstance(ground_raw, torch.Tensor):
                ground_raw = torch.from_numpy(ground_raw.astype(np.float32)).T
            layers = layers_raw.unsqueeze(0).to(device)
            gt     = ground_raw.numpy()
            pred   = ddim_sampler.sample(model, layers, device, eta=0.0, verbose=False)
            pred_windows.append(pred[0].cpu().numpy())
            gt_windows.append(gt)

    total_len  = (len(pred_windows) - 1) * stride + window_size
    pred_full  = hann_overlap_add(pred_windows, window_size, stride, total_len) * scale
    gt_full    = hann_overlap_add(gt_windows,   window_size, stride, total_len) * scale
    pred_full -= pred_full.mean(axis=1, keepdims=True)
    gt_full   -= gt_full.mean(axis=1, keepdims=True)
    pred_full  = remove_spikes(pred_full)
    return pred_full, gt_full


def safe_corr(x, y):
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics(pred, gt, fs=100.0, data_type='disp',
                    stride=512, window_size=1024):
    data_type = data_type.lower()
    if data_type not in ['acc', 'disp']:
        raise ValueError("data_type must be 'acc' or 'disp'.")

    T = pred.shape[1]

    rmse    = float(np.sqrt(np.mean((pred - gt) ** 2)))
    nrmse   = float(rmse / (gt.std() + 1e-8))
    corr_ns = safe_corr(pred[0], gt[0])
    corr_ew = safe_corr(pred[1], gt[1])

    peak_pred = float(np.abs(pred).max())
    peak_gt   = float(np.abs(gt).max())
    epeak     = float(abs(peak_pred - peak_gt) / (peak_gt + 1e-8) * 100)

    trim      = min(stride, T // 4)
    pred_trim = pred[:, trim: T - trim] if T > 2 * trim else pred
    gt_trim   = gt[:,   trim: T - trim] if T > 2 * trim else gt
    t_peak_pred = (float(np.unravel_index(
        np.abs(pred_trim).argmax(), pred_trim.shape)[1]) + trim) / fs
    t_peak_gt   = (float(np.unravel_index(
        np.abs(gt_trim).argmax(), gt_trim.shape)[1]) + trim) / fs
    dt_peak = float(abs(t_peak_pred - t_peak_gt))

    peak_type   = 'PGD' if data_type == 'disp' else 'PGA'
    motion_type = 'Displacement (mm)' if data_type == 'disp' else 'Acceleration (g)'

    metrics = {
        'rmse':        rmse,
        'nrmse':       nrmse,
        'corr_ns':     corr_ns,
        'corr_ew':     corr_ew,
        'motion_type': motion_type,
        'peak_type':   peak_type,
        'peak_pred':   peak_pred,
        'peak_gt':     peak_gt,
        'ePeak_%':     epeak,
        'dt_peak_s':   dt_peak,
    }

    # Compute PSa only in acceleration mode.
    if data_type == 'acc':
        periods, Sa_pred_ns = compute_response_spectrum(pred[0], fs)
        _,       Sa_gt_ns   = compute_response_spectrum(gt[0],   fs)
        _,       Sa_pred_ew = compute_response_spectrum(pred[1], fs)
        _,       Sa_gt_ew   = compute_response_spectrum(gt[1],   fs)

        eSa_ns = float(
            np.sum(np.abs(Sa_pred_ns - Sa_gt_ns)) / (np.sum(np.abs(Sa_gt_ns)) + 1e-8) * 100
        )
        eSa_ew = float(
            np.sum(np.abs(Sa_pred_ew - Sa_gt_ew)) / (np.sum(np.abs(Sa_gt_ew)) + 1e-8) * 100
        )

        metrics.update({
            'eSa_ns_%':    eSa_ns,
            'eSa_ew_%':    eSa_ew,
            '_periods':    periods,
            '_Sa_pred_ns': Sa_pred_ns,
            '_Sa_gt_ns':   Sa_gt_ns,
            '_Sa_pred_ew': Sa_pred_ew,
            '_Sa_gt_ew':   Sa_gt_ew,
        })

    return metrics


def detect_key_segment(gt, fs=100.0, energy_thresh=0.15, min_dur=3.0, pad=2.0):
    T    = gt.shape[1]
    twin = int(fs * 0.5)
    env  = np.zeros(T)
    for ch in range(gt.shape[0]):
        sig = gt[ch]
        for i in range(T):
            lo = max(0, i - twin // 2)
            hi = min(T, i + twin // 2 + 1)
            env[i] += np.sqrt(np.mean(sig[lo:hi] ** 2))

    threshold            = env.max() * energy_thresh
    above                = env >= threshold
    best_start, best_len = 0, 0
    cur_start,  cur_len  = 0, 0
    for i, v in enumerate(above):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len   = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    if best_len / fs < min_dur:
        peak_idx   = int(np.abs(gt).argmax() % T)
        half       = int(min_dur * fs / 2)
        best_start = max(0, peak_idx - half)
        best_len   = int(min_dur * fs)

    t_start = max(0.0,     best_start / fs - pad)
    t_end   = min(T / fs, (best_start + best_len) / fs + pad)
    return t_start, t_end


def _plot_fft_ax(ax, gt_ch, pred_ch, T, fs, direction, C_GT, C_PRED, ALPHA):
    """Plot the Fourier amplitude spectrum subplot."""
    freqs    = np.fft.rfftfreq(T, d=1.0 / fs)
    fft_gt   = np.abs(np.fft.rfft(gt_ch))   / T
    fft_pred = np.abs(np.fft.rfft(pred_ch)) / T
    ax.semilogy(freqs, fft_gt,   color=C_GT,   alpha=ALPHA, lw=1.2, label='GT')
    ax.semilogy(freqs, fft_pred, color=C_PRED, lw=1.1, ls='--',    label='Pred')
    pos_vals = np.concatenate([fft_gt[fft_gt > 0], fft_pred[fft_pred > 0]])
    ymin = max(1e-6, pos_vals.min() * 0.1) if pos_vals.size > 0 else 1e-6
    ymax = max(fft_gt.max(), fft_pred.max()) * 3.0 if pos_vals.size > 0 else 1.0
    ax.set_title(f'{direction}  Fourier Amplitude Spectrum')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim([0, 25])
    ax.set_ylim([ymin, ymax])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.25)


def _plot_spectrum_ax(ax, periods, Sa_gt, Sa_pred, eSa, direction,
                      C_GT, C_PRED, ALPHA):
    """Plot the PSa response spectrum subplot, used only in acc mode."""
    ax.plot(periods, Sa_gt,   color=C_GT,   alpha=ALPHA, lw=1.5, label='GT PSa')
    ax.plot(periods, Sa_pred, color=C_PRED, lw=1.3, ls='--',     label='Pred PSa')
    ax.fill_between(periods, Sa_gt, Sa_pred, alpha=0.12, color=C_PRED)
    ax.set_xlabel('Period (s)', fontsize=11)
    ax.set_ylabel('PSa (g)',    fontsize=11)
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    ax.set_title(f'{direction}  PSa Spectrum   eSa={eSa:.1f}%', fontsize=13)


def plot_record(fname, pred, gt, metrics, fs, save_dir):
    plt.rcParams.update({
        'font.weight':      'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelsize':   13,
        'axes.titlesize':   13,
        'xtick.labelsize':  11,
        'ytick.labelsize':  11,
        'legend.fontsize':  11,
        'figure.titlesize': 13,
        'font.size':        12,
    })

    data_type   = 'acc' if metrics['peak_type'] == 'PGA' else 'disp'
    T           = pred.shape[1]
    time        = np.arange(T) / fs
    t_start, t_end = detect_key_segment(gt, fs=fs)
    i_start     = int(t_start * fs)
    i_end       = int(t_end   * fs)
    time_zoom   = time[i_start:i_end]

    # disp uses a 3-row layout; acc uses a 4-row layout.
    n_rows   = 3 if data_type == 'disp' else 4
    fig_h    = 12 if data_type == 'disp' else 16
    fig      = plt.figure(figsize=(20, fig_h))
    gs_fig   = gridspec.GridSpec(n_rows, 2, hspace=0.55, wspace=0.28)

    C_GT    = '#1A1A2E'
    C_PRED  = '#D95319'
    C_SHADE = '#FFF3E0'
    C_BOX   = '#FF6F00'
    ALPHA   = 0.75
    motion_type = metrics['motion_type']

    for col, (ch, d) in enumerate(zip([0, 1], ['NS', 'EW'])):

        # Row 0: full time history.
        ax0 = fig.add_subplot(gs_fig[0, col])
        ax0.plot(time, gt[ch],   color=C_GT,   alpha=ALPHA, lw=1.2, label='Ground Truth')
        ax0.plot(time, pred[ch], color=C_PRED, lw=1.1, ls='--',    label='AI Prediction')
        ax0.axvspan(t_start, t_end, color=C_SHADE, alpha=0.6, zorder=0)
        y_lo, y_hi = ax0.get_ylim()
        ax0.add_patch(plt.Rectangle(
            (t_start, y_lo), t_end - t_start, y_hi - y_lo,
            linewidth=1.5, edgecolor=C_BOX, facecolor='none', zorder=3, linestyle='--'
        ))
        ax0.annotate('Key Segment ↓', xy=((t_start + t_end) / 2, y_hi * 0.88),
                     ha='center', fontsize=11, color=C_BOX,
                     fontstyle='italic', fontweight='bold')
        ax0.set_title(f'{d}  Full Time History')
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel(motion_type)
        ax0.legend(loc='upper right')
        ax0.grid(True, alpha=0.25)

        # Row 1: key-segment zoom.
        ax1 = fig.add_subplot(gs_fig[1, col])
        ax1.plot(time_zoom, gt[ch][i_start:i_end],
                 color=C_GT,   alpha=ALPHA, lw=1.8, label='Ground Truth', zorder=3)
        ax1.plot(time_zoom, pred[ch][i_start:i_end],
                 color=C_PRED, lw=1.5, ls='--', label='AI Prediction', zorder=4)
        peak_idx_gt   = np.argmax(np.abs(gt[ch][i_start:i_end]))
        peak_idx_pred = np.argmax(np.abs(pred[ch][i_start:i_end]))
        ax1.scatter(time_zoom[peak_idx_gt],
                    gt[ch][i_start:i_end][peak_idx_gt],
                    color=C_GT,   s=80, zorder=5, marker='*', label='GT Peak')
        ax1.scatter(time_zoom[peak_idx_pred],
                    pred[ch][i_start:i_end][peak_idx_pred],
                    color=C_PRED, s=80, zorder=5, marker='*', label='Pred Peak')
        local_pcc = safe_corr(gt[ch][i_start:i_end], pred[ch][i_start:i_end])
        ax1.text(0.02, 0.96, f'Local PCC = {local_pcc:.3f}',
                 transform=ax1.transAxes, fontsize=11, va='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        ax1.set_title(f'{d}  Key Segment  [{t_start:.1f}s – {t_end:.1f}s]')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(motion_type)
        ax1.legend(loc='upper right', ncol=2)
        ax1.grid(True, alpha=0.25)

        for x_conn in [t_start, t_end]:
            fig.add_artist(ConnectionPatch(
                xyA=(x_conn, ax0.get_ylim()[0]), coordsA=ax0.transData,
                xyB=(x_conn, ax1.get_ylim()[1]), coordsB=ax1.transData,
                color=C_BOX, lw=0.8, ls=':', alpha=0.6
            ))

        if data_type == 'acc':
            # Row 2: PSa response spectrum.
            ax2 = fig.add_subplot(gs_fig[2, col])
            Sa_gt_arr   = metrics['_Sa_gt_ns']   if ch == 0 else metrics['_Sa_gt_ew']
            Sa_pred_arr = metrics['_Sa_pred_ns'] if ch == 0 else metrics['_Sa_pred_ew']
            eSa         = metrics['eSa_ns_%']    if ch == 0 else metrics['eSa_ew_%']
            _plot_spectrum_ax(ax2, metrics['_periods'],
                              Sa_gt_arr, Sa_pred_arr, eSa, d,
                              C_GT, C_PRED, ALPHA)
            # Row 3: Fourier spectrum.
            ax3 = fig.add_subplot(gs_fig[3, col])
            _plot_fft_ax(ax3, gt[ch], pred[ch], T, fs, d, C_GT, C_PRED, ALPHA)
        else:
            # Row 2: Fourier spectrum. PSa is not used in disp mode.
            ax2 = fig.add_subplot(gs_fig[2, col])
            _plot_fft_ax(ax2, gt[ch], pred[ch], T, fs, d, C_GT, C_PRED, ALPHA)

    plt.savefig(os.path.join(save_dir, f'{fname}_full.png'), dpi=150, bbox_inches='tight')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def plot_summary(all_metrics, save_dir, data_type='disp'):
    if not all_metrics:
        return

    fnames    = [m['filename']  for m in all_metrics]
    peak_gt   = np.array([m['peak_gt']   for m in all_metrics])
    peak_pred = np.array([m['peak_pred'] for m in all_metrics])
    rmse_arr  = np.array([m['rmse']      for m in all_metrics])
    corr_ns   = np.array([m['corr_ns']   for m in all_metrics])
    corr_ew   = np.array([m['corr_ew']   for m in all_metrics])
    epeak_arr = np.array([m['ePeak_%']   for m in all_metrics])

    peak_label = 'PGD' if data_type == 'disp' else 'PGA'
    fname_out  = f'summary_p{"gd" if data_type == "disp" else "ga"}_scatter.png'

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(peak_gt.max(), peak_pred.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1.0, alpha=0.5, label='1:1 line')
    ax.scatter(peak_gt, peak_pred, c='#D95319', s=60, alpha=0.8, zorder=3)
    for i, fn in enumerate(fnames):
        ax.annotate(fn.replace('fzcgznRun', 'R'), (peak_gt[i], peak_pred[i]),
                    fontsize=6, alpha=0.6, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel(f'True {peak_label}',      fontsize=12)
    ax.set_ylabel(f'Predicted {peak_label}', fontsize=12)
    ax.set_title(f'{peak_label} Scatter Plot  (All Validation Records)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname_out), dpi=150)
    plt.close()

    def _hist(ax, data, xlabel, color, title):
        ax.hist(data, bins=8, color=color, alpha=0.8, edgecolor='white')
        ax.axvline(data.mean(), color='k', lw=1.5, ls='--',
                   label=f'Mean={data.mean():.2f}')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Count',  fontsize=11)
        ax.set_title(title,     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if data_type == 'acc':
        eSa_ns = np.array([m['eSa_ns_%'] for m in all_metrics])
        eSa_ew = np.array([m['eSa_ew_%'] for m in all_metrics])
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Error Distribution  (All Validation Records)', fontsize=13)
        _hist(axs[0, 0], rmse_arr,             'RMSE',               '#2196F3', 'RMSE Distribution')
        _hist(axs[0, 1], (corr_ns+corr_ew)/2,  'PCC (avg)',          '#4CAF50', 'PCC Distribution')
        _hist(axs[0, 2], epeak_arr,            f'e{peak_label} (%)', '#FF9800', f'e{peak_label} Distribution')
        _hist(axs[1, 0], eSa_ns,               'eSa NS (%)',         '#9C27B0', 'eSa NS Distribution')
        _hist(axs[1, 1], eSa_ew,               'eSa EW (%)',         '#673AB7', 'eSa EW Distribution')
        _hist(axs[1, 2], (eSa_ns+eSa_ew)/2,   'eSa avg (%)',        '#E91E63', 'eSa Avg Distribution')
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Error Distribution  (All Validation Records)', fontsize=13)
        _hist(axs[0], rmse_arr,            'RMSE',               '#2196F3', 'RMSE Distribution')
        _hist(axs[1], (corr_ns+corr_ew)/2, 'PCC (avg)',          '#4CAF50', 'PCC Distribution')
        _hist(axs[2], epeak_arr,           f'e{peak_label} (%)', '#FF9800', f'e{peak_label} Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_error_dist.png'), dpi=150)
    plt.close()
    print(f"[Summary] Summary figures saved to: {save_dir}")


def reconstruct_all(model, ddim_sampler, val_ds, device,
                    save_dir, fs=100.0,
                    window_size=1024, stride=512,
                    data_type='disp'):
    data_type = data_type.lower()
    if data_type not in ['acc', 'disp']:
        raise ValueError("data_type must be 'acc' or 'disp'.")

    out_dir = os.path.join(save_dir, 'full_record_results')
    os.makedirs(out_dir, exist_ok=True)

    record_map = {}
    idx = 0
    for meta in val_ds.target_files:
        fname    = meta['filename'].replace('.npz', '')
        npz_path = os.path.join(val_ds.root_dir, meta['folder'], fname + '.npz')
        try:
            data      = np.load(npz_path, allow_pickle=True)
            total_len = data['labels'].shape[0]
            n_win     = max(1, (total_len - window_size) // stride + 1)
        except Exception:
            n_win = 1
        record_map[fname] = [
            val_ds.samples[idx + i]
            for i in range(n_win)
            if idx + i < len(val_ds.samples)
        ]
        idx += n_win

    print(f"\n[ReconstructAll] Reconstructing {len(record_map)} validation records...")

    all_metrics = []
    scale       = val_ds.scale

    for fname, samples in tqdm(record_map.items(), desc='Reconstructing'):
        if not samples:
            continue
        pred_full, gt_full = invert_record(
            model, ddim_sampler, samples, device,
            window_size=window_size, stride=stride, scale=scale
        )
        metrics             = compute_metrics(
            pred_full, gt_full, fs=fs,
            data_type=data_type, stride=stride, window_size=window_size
        )
        metrics['filename'] = fname
        all_metrics.append(metrics)
        plot_record(fname, pred_full, gt_full, metrics, fs, out_dir)

    peak_err_label = 'ePGD(%)' if data_type == 'disp' else 'ePGA(%)'

    # CSV fields depend on data_type.
    csv_fields = [
        'filename', 'rmse', 'nrmse', 'corr_ns', 'corr_ew',
        'motion_type', 'peak_type', 'peak_gt', 'peak_pred',
        'ePeak_%', 'dt_peak_s',
    ]
    if data_type == 'acc':
        csv_fields += ['eSa_ns_%', 'eSa_ew_%']

    csv_path = os.path.join(out_dir, 'metrics_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"  CSV saved to: {csv_path}")

    try:
        plot_summary(all_metrics, out_dir, data_type=data_type)
    except Exception as e:
        print(f"\nWarning: plot_summary failed, but the CSV has been saved. Reason: {repr(e)}")

    def _stat(key):
        arr = np.array([m[key] for m in all_metrics], dtype=np.float64)
        return arr.mean(), arr.std(), arr.min(), arr.max()

    print(f"\n{'=' * 60}")
    print(f"  Full evaluation results ({len(all_metrics)} records)")
    print(f"  {'Metric':<18} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-' * 58}")

    stat_items = [
        ('rmse',      'RMSE'),
        ('nrmse',     'NRMSE'),
        ('corr_ns',   'CorrNS'),
        ('corr_ew',   'CorrEW'),
        ('ePeak_%',   peak_err_label),
        ('dt_peak_s', 'ΔtPeak(s)'),
    ]
    if data_type == 'acc':
        stat_items += [
            ('eSa_ns_%', 'eSa NS(%)'),
            ('eSa_ew_%', 'eSa EW(%)'),
        ]

    for key, label in stat_items:
        mu, std, mn, mx = _stat(key)
        print(f"  {label:<18} {mu:>8.4f}  {std:>8.4f}  {mn:>8.4f}  {mx:>8.4f}")
    print(f"{'=' * 60}")
    print(f"  CSV:    {csv_path}")
    print(f"  Images: {out_dir}\n")

    return all_metrics