import os, glob, re
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- CONFIG (no device here) --------------------
DATA_ROOT   =    # add input files/folder path
CHANNEL_DIR = ""           # e.g. "" or "C4-M1"
CHANNEL_IN_FILENAME = None # optional substring to filter filenames

SEQ_LEN        = 10
STRIDE_EPOCHS  = 5
IMG_H, IMG_W   =           # (H, W)-- write give the image size here

CLASS_LABELS = {'W':0, 'N1':1, 'N2':2, 'N3':3, 'R':4}
INV_LABELS   = {v:k for k,v in CLASS_LABELS.items()}
NUM_CLASSES  = len(CLASS_LABELS)

# -------------------- UTILS --------------------
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_subjects(root):
    if not os.path.isdir(root):
        return []
    return sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
        key=natural_key
    )

def list_all_subject_paths(data_root):
    """
    Returns [(subject_id, subject_path), ...]
    Works with either:
      - root/S001/W/*.png, ...
      - or root/Training_Set/S001/W/*.png etc.
    """
    candidates = []
    split_dirs = [
        os.path.join(data_root, "Training_Set"),
        os.path.join(data_root, "Validation_Set"),
        os.path.join(data_root, "Testing_Set"),
    ]
    any_split = any(os.path.isdir(d) for d in split_dirs)

    if any_split:
        for sd in split_dirs:
            if os.path.isdir(sd):
                for s in list_subjects(sd):
                    candidates.append((s, os.path.join(sd, s)))
    else:
        for s in list_subjects(data_root):
            candidates.append((s, os.path.join(data_root, s)))

    seen, unique = set(), []
    for sid, spath in candidates:
        if sid not in seen:
            unique.append((sid, spath))
            seen.add(sid)
    return unique

def collect_subject_items(subj_path):
    """
    Returns a list of (filepath, label_id) for one subject.
    """
    items = []
    for stage, lab in CLASS_LABELS.items():
        stage_dir = os.path.join(subj_path, stage) if CHANNEL_DIR == "" \
                    else os.path.join(subj_path, stage, CHANNEL_DIR)
        if not os.path.isdir(stage_dir):
            continue

        files = [f for f in glob.glob(os.path.join(stage_dir, "*.*"))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]

        if CHANNEL_IN_FILENAME:
            files = [f for f in files if CHANNEL_IN_FILENAME in os.path.basename(f)]

        files = sorted(
            files,
            key=lambda fp: natural_key(os.path.join(os.path.basename(os.path.dirname(fp)),
                                                    os.path.basename(fp)))
        )
        items += [(fp, lab) for fp in files]

    items = sorted(items, key=lambda x: natural_key(os.path.basename(x[0])))
    return items

def make_sequences_from_slice(items, start, end, seq_len=SEQ_LEN, stride=STRIDE_EPOCHS):
    seqs = []
    if end - start < seq_len:
        return seqs
    for i in range(start, end - seq_len + 1, stride):
        chunk = items[i:i+seq_len]
        if len(chunk) == seq_len:
            seqs.append(chunk)
    return seqs

def make_all_sequences(items, seq_len=SEQ_LEN, stride=STRIDE_EPOCHS):
    return make_sequences_from_slice(items, 0, len(items), seq_len=seq_len, stride=stride)

def compute_has_flags(items):
    """
    Returns the (hasN3, hasR) flags for a single subject based on its item labels.
    """
    labels = [lab for _, lab in items]
    hasN3 = int(CLASS_LABELS['N3'] in labels)
    hasR  = int(CLASS_LABELS['R']  in labels)
    return hasN3, hasR

def save_plot(xs, ys_list, labels, title, path):
    plt.figure()
    for ys, lbl in zip(ys_list, labels):
        plt.plot(xs, ys, label=lbl)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(labels[0].split()[-1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_confmat(cm, labels, title, path):
    plt.figure()
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    perc = (cm / row_sums) * 100.0
    annot = np.empty_like(cm).astype(object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n{perc[i, j]:.1f}%"
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _flat_paths(seqs):
    return [p for seq in seqs for (p, _) in seq]

def assert_disjoint(a, b, name_a="A", name_b="B"):
    A, B = set(_flat_paths(a)), set(_flat_paths(b))
    inter = A & B
    print(f"[CHECK] overlap {name_a}∩{name_b}: {len(inter)}")
    assert len(inter) == 0, f"Leakage: {name_a} and {name_b} share {len(inter)} frames!"

def _letterbox_to_multiple(img: Image.Image, mult=32, mode="down"):
    """
    Optional helper if you ever want letterboxed resize to multiples of `mult`.
    Currently unused when RESIZE_MODE == 'fixed'.
    """
    w, h = img.size
    if mode == "down":
        target_w = (w // mult) * mult
        target_h = (h // mult) * mult
    else:
        target_w = ((w + mult - 1) // mult) * mult
        target_h = ((h + mult - 1) // mult) * mult

    scale = min(target_w / max(1, w), target_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    left   = pad_w // 2
    right  = pad_w - left
    top    = pad_h // 2
    bottom = pad_h - top
    if pad_w > 0 or pad_h > 0:
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
    return img

# -------------------- DATASET --------------------
class ModSpecSeqDataset(Dataset):
    """
    Dataset of sequences of modulation spectrogram images.
    Each item is:
        x: [T, C, H, W]
        y: [T] (class indices)
        subj_id: string subject identifier
    """
    def __init__(self, seqs, subj_ids):
        self.seqs = seqs
        self.subj_ids = subj_ids

    def __len__(self):
        return len(self.seqs)

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_W, IMG_H), Image.BICUBIC)  # (W, H)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # [C, H, W]
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        xs, ys = [], []
        for p, lab in seq:
            xs.append(self._load_img(p))
            ys.append(lab)
        x = torch.stack(xs, dim=0)          # [T, C, H, W]
        y = torch.tensor(ys).long()         # [T]
        return x, y, self.subj_ids[idx]

# -------------------- MODEL --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, drop=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()
        self.do   = nn.Dropout2d(drop)
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.act(x); x = self.do(x); return x

class ResBlock(nn.Module):
    def __init__(self, ch, drop=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.act   = nn.GELU()
        self.do    = nn.Dropout2d(drop)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
    def forward(self, x):
        out = self.conv1(x); out = self.bn1(out); out = self.act(out); out = self.do(out)
        out = self.conv2(out); out = self.bn2(out)
        out = self.act(out + x)
        return out

class EEGSNetLike(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, drop=0.5):
        super().__init__()
        self.c1 = ConvBlock(3, 32, 3, 1, 1, drop)
        self.r1 = ResBlock(32, drop)
        self.c2 = ConvBlock(32, 64, 3, 2, 1, drop)
        self.r2 = ResBlock(64, drop)
        self.c3 = ConvBlock(64, 128, 3, 2, 1, drop)
        self.r3 = ResBlock(128, drop)
        self.c4 = ConvBlock(128, 128, 3, 1, 1, drop)
        self.r4 = ResBlock(128, drop)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = 128

        self.aux_head = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, n_classes)
        )

        self.bilstm = nn.LSTM(
            input_size=self.feat_dim, hidden_size=128, num_layers=2,
            batch_first=True, dropout=drop, bidirectional=True
        )
        self.main_head = nn.Linear(128 * 2, n_classes)

    def forward(self, x, train_mode=True):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        h = self.c1(x); h = self.r1(h)
        h = self.c2(h); h = self.r2(h)
        h = self.c3(h); h = self.r3(h)
        h = self.c4(h); h = self.r4(h)
        h = self.pool(h).view(B*T, -1)
        aux_logits = self.aux_head(h)
        h = h.view(B, T, -1)
        out, _ = self.bilstm(h)
        main_logits = self.main_head(out)
        return main_logits, aux_logits.view(B, T, -1)

# -------------------- NORMALIZATION / CLASS WEIGHTS --------------------
def zscore_from_loader(loader, device):
    m = None; s = None; n = 0
    for x, y, _ in loader:
        bs, t, c, h, w = x.shape
        x = x.view(bs*t, c, h, w).float().to(device)
        x_mean = x.mean(dim=(0, 2, 3))
        x_sq   = (x**2).mean(dim=(0, 2, 3))
        if m is None:
            m = x_mean * (bs*t)
            s = x_sq   * (bs*t)
        else:
            m += x_mean * (bs*t)
            s += x_sq   * (bs*t)
        n += (bs*t)
    mean = (m / max(n, 1))
    std  = (s / max(n, 1) - mean**2).clamp(min=1e-12).sqrt()
    return mean.detach(), std.detach()

def apply_norm(x, mean, std):
    return (x - mean[None, None, :, None, None]) / std[None, None, :, None, None]

def get_class_weights(loaders):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for loader in loaders:
        for _, y, _ in loader:
            y_flat = y.view(-1).numpy()
            for c in range(NUM_CLASSES):
                counts[c] += np.sum(y_flat == c)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
