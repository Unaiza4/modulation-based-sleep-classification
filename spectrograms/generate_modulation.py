# -*- coding: utf-8 -*-
"""
Batch SAFE modulation spectrograms for sleep staging:
- Accepts a single CSV path OR a directory containing many CSVs
- Native FS (no resample)
- Tries 60 s window (2×30 s) if label-consistent; otherwise falls back to 30 s
- Mild DC suppression (≤0.1 Hz × 0.4) to improve visibility
- Global robust normalization (6–99.6 percentiles)
- **Outputs only 152x120** colored PNGs (no axes/titles/colorbar)
- Folder layout: <OUT>/<subject>/<stage>/<image>.png
- Skips P/blank/NaN stages
"""

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from scipy.signal import butter, filtfilt, hilbert
from matplotlib import cm

# ------------------- INPUT -------------------
INPUT_PATH = r""      # add your file/folder path here, e.g. r"C:\data\csvs"

# Column names & sampling
CHANNEL   = "C4-M1"
FS        = 100.0            # native sampling rate 
EPOCH_SEC = 30.0

# Modulation window: attempt longer context but guard label integrity
MOD_WINDOW_EPOCHS = 2       
AGREE_TAU = 0.75             
SKIP_IF_FAIL = False         

# Carrier band
BAND_CFG = {
    "delta": {"range": (0.5, 4.0),  "step": 0.5},
    "theta": {"range": (4.0, 8.0),  "step": 0.5},
    "alpha": {"range": (8.0, 12.0), "step": 0.5},
    "sigma": {"range": (12.0,15.0), "step": 0.5},
    "beta":  {"range": (15.0,30.0), "step": 1.0},
}

# Envelope chain for modulation
ENV_LPF_HZ = 10.0   
ENV_FS     = 20.0   
MOD_F_MAX  = 8.0    

# Mild DC suppression to avoid dominance of near-0 Hz modulation
DC_SUPPRESS_HZ   = 0.1
DC_SUPPRESS_GAIN = 0.4

# Normalization (global, robust)
PCT_LOW, PCT_HIGH = 6, 99.6

# ------------------- OUTPUT -------------------
OUT_BIG = r""         # add your output base folder here
BIG_HW  = (152, 120)  # (H, W)
CMAP_NAME = "viridis"
# ------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def butter_bp(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    ny = fs / 2.0
    lo_n, hi_n = max(lo / ny, 1e-6), min(hi / ny, 0.999999)
    b, a = butter(order, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, x)

def butter_lp(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    ny = fs / 2.0
    fc_n = min(fc / ny, 0.999999)
    b, a = butter(order, fc_n, btype='low')
    return filtfilt(b, a, x)

def robust_normalize_global(M: np.ndarray, p_lo=PCT_LOW, p_hi=PCT_HIGH, eps=1e-8) -> np.ndarray:
    if not np.isfinite(M).any():
        return np.zeros_like(M, dtype=np.float32)
    finite = M[np.isfinite(M)]
    lo, hi = np.percentile(finite, p_lo), np.percentile(finite, p_hi)
    if not np.isfinite(lo): lo = np.nanmin(M)
    if not np.isfinite(hi): hi = np.nanmax(M)
    rng = max(hi - lo, eps)
    return np.clip((M - lo) / rng, 0, 1).astype(np.float32)

def split_band_edges(lo: float, hi: float, step: Optional[float]) -> List[Tuple[float, float]]:
    if step is None or step <= 0:
        return [(lo, hi)]
    edges = list(np.arange(lo, hi, step))
    if len(edges) == 0 or not math.isclose(edges[-1], hi):
        edges.append(hi)
    return [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

def build_carrier_bins(band_cfg: dict) -> List[Tuple[float, float]]:
    bins = []
    for _, cfg in band_cfg.items():
        lo, hi = cfg["range"]
        step = cfg.get("step", None)
        bins.extend(split_band_edges(lo, hi, step))
    return bins

def modulation_map_pure(
    x: np.ndarray,
    fs: float,
    carrier_bins: List[Tuple[float, float]],
    env_lpf_hz: float,
    env_fs: float,
    mod_f_max: float,
    log1p: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    decim = int(round(fs / env_fs))
    decim = max(decim, 1)

    cols = []
    f_mod_keep = None
    for (lo, hi) in carrier_bins:
        xb = butter_bp(x, fs, lo, hi, order=4)
        env = np.abs(hilbert(xb))
        env = butter_lp(env, fs, env_lpf_hz, order=4)
        env_ds = env[::decim] if decim > 1 else env
        fs_env = fs / decim

        if env_ds.size < 4:
            cols.append(np.zeros(1, dtype=np.float64))
            if f_mod_keep is None:
                f_mod_keep = np.array([0.0])
            continue

        M = np.abs(np.fft.rfft(env_ds)) ** 2
        f_mod = np.fft.rfftfreq(env_ds.size, d=1 / fs_env)
        keep = f_mod <= mod_f_max
        if not np.any(keep):
            keep = slice(None)
        cols.append(M[keep])
        f_mod_keep = f_mod[keep]

    # Align column lengths if needed
    max_len = max(len(c) for c in cols)
    cols = [np.pad(c, (0, max_len - len(c))) for c in cols]
    P = np.stack(cols, axis=1).astype(np.float64)  # [F_mod x F_carrier]

    if log1p:
        P = np.log1p(P)

    # Mild DC suppression
    if f_mod_keep is not None:
        dc_mask = (f_mod_keep <= DC_SUPPRESS_HZ)
        if np.any(dc_mask):
            P[dc_mask, :] *= DC_SUPPRESS_GAIN

    Pnorm = robust_normalize_global(P, PCT_LOW, PCT_HIGH)
    return Pnorm, f_mod_keep

def save_color(arr01: np.ndarray, out_hw: Tuple[int, int], out_path: str, cmap_name: str = CMAP_NAME):
    cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(np.clip(arr01, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((out_hw[1], out_hw[0]), resample=Image.BILINEAR)  # (W,H)
    img.save(out_path)

def normalize_stage(s: str) -> Optional[str]:
    if s is None:
        return None
    s2 = str(s).strip().upper()
    if s2 in {"", "P", "NAN"}:
        return None
    allowed = {"W", "N1", "N2", "N3", "R"}
    return s2 if s2 in allowed else None

def epoch_label_array(stage_series, samples_per_epoch):
    n_ep = len(stage_series) // samples_per_epoch
    labs = stage_series[: n_ep * samples_per_epoch: samples_per_epoch]
    return np.array([normalize_stage(s) for s in labs], dtype=object)

def window_epoch_span(center_epoch, total_epochs, k_epochs):
    if k_epochs <= 1:
        return center_epoch, center_epoch + 1
    half = k_epochs // 2
    if k_epochs % 2 == 1:  
        s = center_epoch - half
        e = center_epoch + half + 1
    else:                    
        s = center_epoch - half
        e = center_epoch + (half - 1) + 1
    s = max(0, s)
    e = min(total_epochs, e)
    return s, e

def window_label_agreement(epoch_labels_np, center_epoch, k_epochs, agree_tau=0.75):
    total_epochs = len(epoch_labels_np)
    s_ep, e_ep = window_epoch_span(center_epoch, total_epochs, k_epochs)
    win_labs = epoch_labels_np[s_ep:e_ep]
    win_labs = [l for l in win_labs if l is not None]
    if len(win_labs) == 0:
        return False, (s_ep, e_ep)
    center_lab = epoch_labels_np[center_epoch]
    match = sum(1 for l in win_labs if l == center_lab)
    frac = match / len(win_labs)
    return (frac >= agree_tau), (s_ep, e_ep)

def pick_window_by_span(sig, samples_per_epoch, span):
    s_ep, e_ep = span
    s = s_ep * samples_per_epoch
    e = e_ep * samples_per_epoch
    return sig[s:e]

# ------------------- BATCH HANDLER -------------------

def list_csvs(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return [p]
    if p.is_dir():
        files = sorted(Path(p).glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
        return files
    raise FileNotFoundError(f"Path not found: {input_path}")

def process_csv(csv_path: Path, carrier_bins: List[Tuple[float, float]]):
    print(f"\n--- Processing: {csv_path} ---")
    df = pd.read_csv(csv_path)
    if CHANNEL not in df.columns:
        print(f"[SKIP] Channel '{CHANNEL}' not found in {csv_path.name}")
        return (0, 0, 0) 

    if "Sleep_Stage" not in df.columns:
        print(f"[SKIP] 'Sleep_Stage' column not found in {csv_path.name}")
        return (0, 0, 0)

    sig = df[CHANNEL].to_numpy(dtype=float)
    stages = df["Sleep_Stage"].to_numpy()

    samples_per_epoch = int(round(FS * EPOCH_SEC))
    num_epochs = len(sig) // samples_per_epoch
    if num_epochs < 1:
        print(f"[SKIP] Not enough samples for 30 s epochs in {csv_path.name}")
        return (0, 0, 0)

    sig = sig[: num_epochs * samples_per_epoch]
    stages = stages[: num_epochs * samples_per_epoch]

    epoch_labels_np = epoch_label_array(stages, samples_per_epoch)
    subject_id = csv_path.stem.split("_PSG")[0]

    saved_big = 0
    n_fallback = n_skipped = 0

    n_60s = 0
    n_30s = 0

    for i in range(num_epochs):
        stage = epoch_labels_np[i]
        if stage is None:
            continue  # skip P/blank/NaN/unknown

        ok, span = window_label_agreement(epoch_labels_np, i, MOD_WINDOW_EPOCHS, AGREE_TAU)

        if ok:
            # 60 s OK only if exactly 2 epochs spanned; else treat as not OK
            s_ep, e_ep = span
            if (e_ep - s_ep) == MOD_WINDOW_EPOCHS:
                x_win = pick_window_by_span(sig, samples_per_epoch, span)
                n_60s += 1  # 60 s
            else:
                ok = False  # will fall back
        if not ok:
            if SKIP_IF_FAIL:
                n_skipped += 1
                continue
            x_win = pick_window_by_span(sig, samples_per_epoch, (i, i+1))  # 30 s fallback
            n_fallback += 1
            n_30s += 1

        if x_win.size < samples_per_epoch:  # guard (covers odd edge cases)
            n_skipped += 1
            continue

        x_win = butter_bp(x_win, FS, 0.3, 40.0, order=4)
        x_win = (x_win - x_win.mean()) / (x_win.std() + 1e-8)

        P01, _ = modulation_map_pure(
            x_win, FS, carrier_bins,
            env_lpf_hz=ENV_LPF_HZ,
            env_fs=ENV_FS,
            mod_f_max=MOD_F_MAX,
            log1p=True
        )

        # ---- SAVE Spectrograms----
        out_dir_big = os.path.join(OUT_BIG, subject_id, stage)
        ensure_dir(out_dir_big)
        base = f"{subject_id}_{CHANNEL}_epoch{i+1:03d}.png"
        save_color(P01, BIG_HW, os.path.join(out_dir_big, base), cmap_name=CMAP_NAME)
        saved_big += 1

    print(f"Saved {BIG_HW[0]}x{BIG_HW[1]}: {saved_big} | fallbacks: {n_fallback} | skipped: {n_skipped}")
    print(f"60 s windows: {n_60s} | 30 s windows: {n_30s}")
    return (saved_big, n_fallback, n_skipped)

def main():
    # Build carrier bins once
    carrier_bins = build_carrier_bins(BAND_CFG)

    # Expand INPUT_PATH into a list of CSV files
    csv_files = list_csvs(INPUT_PATH)
    print(f"Found {len(csv_files)} CSV file(s).")

    total_big = total_fallback = total_skipped = 0

    for csv_path in csv_files:
        try:
            s_big, fb, sk = process_csv(csv_path, carrier_bins)
            total_big     += s_big
            total_fallback += fb
            total_skipped  += sk
        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")

    print("\n=== Batch Summary ===")
    print(f"Total saved {BIG_HW[0]}x{BIG_HW[1]}: {total_big}")
    print(f"Total fallbacks    : {total_fallback}")
    print(f"Total skipped      : {total_skipped}")
    print("Done.")

if __name__ == "__main__":
    main()
