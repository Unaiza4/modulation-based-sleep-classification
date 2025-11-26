import os, glob, re, math, json, random, shutil, time
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
cudnn.benchmark = True
cudnn.deterministic = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = (DEVICE.type == "cuda")
NON_BLOCKING = (DEVICE.type == "cuda")
USE_AMP = (DEVICE.type == "cuda")
# -------------------- CONFIG --------------------
# Root that contains Training_Set / Validation_Set / Testing_Set
DATA_ROOT   = r   # your path
RESULTS_DIR = r  # output directory

# Channel/filename filtering
CHANNEL_DIR = ""              # no subfolder for channel
CHANNEL_IN_FILENAME = "C4-M1"  # keep only files containing this; set to None to disable

SEQ_LEN        = 10
STRIDE_EPOCHS  = 5
BATCH_SIZE     = 2
MAX_EPOCHS     = 150
VAL_EVERY      = 3
PATIENCE       = 10
DROPOUT_P      = 0.5
AUX_WEIGHT     = 0.5          # for CNN-only we set aux_logits = main_logits, so this is effectively 1.5x CE
                             # you can set this to 0.0 if you want pure single-head loss
IMG_H, IMG_W   = 120, 152
RESIZE_MODE    = "fixed"      
MULTIPLE_OF    = 32           # kept for other modes
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 4
SEED           = 42

VAL_SUBJ_FRACTION = 0.15
N_FOLDS      = 5
ACCUM_STEPS  = 1

CLASS_LABELS = {'W':0, 'N1':1, 'N2':2, 'N3':3, 'R':4}
INV_LABELS   = {v:k for k,v in CLASS_LABELS.items()}
NUM_CLASSES  = len(CLASS_LABELS)

os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# -------------------- UTIL --------------------
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

    seen = set()
    unique = []
    for sid, spath in candidates:
        if sid not in seen:
            unique.append((sid, spath))
            seen.add(sid)
    return unique

def collect_subject_items(subj_path):
    items = []
    for stage, lab in CLASS_LABELS.items():
        if CHANNEL_DIR == "":
            stage_dir = os.path.join(subj_path, stage)
        else:
            stage_dir = os.path.join(subj_path, stage, CHANNEL_DIR)

        if not os.path.isdir(stage_dir):
            continue

        files = [
            f for f in glob.glob(os.path.join(stage_dir, "*.*"))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]

        if CHANNEL_IN_FILENAME:
            files = [f for f in files if CHANNEL_IN_FILENAME in os.path.basename(f)]

        files = sorted(
            files,
            key=lambda fp: natural_key(
                os.path.join(os.path.basename(os.path.dirname(fp)), os.path.basename(fp))
            )
        )
        items += [(fp, lab) for fp in files]

    # Sort by filename for stable temporal order
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
    w, h = img.size  # (W,H)
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
    def __init__(self, seqs, subj_ids):
        self.seqs = seqs
        self.subj_ids = subj_ids

    def __len__(self):
        return len(self.seqs)

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        if RESIZE_MODE == "fixed":
            img = img.resize((IMG_W, IMG_H), Image.BICUBIC)
        elif RESIZE_MODE == "letterbox32_down":
            img = _letterbox_to_multiple(img, mult=MULTIPLE_OF, mode="down")
        elif RESIZE_MODE == "native":
            pass
        else:
            raise ValueError(f"Unknown RESIZE_MODE={RESIZE_MODE}")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # [C,H,W]
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

# -------------------- MODEL: CNN-ONLY --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, drop=DROPOUT_P):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()
        self.do   = nn.Dropout2d(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, ch, drop=DROPOUT_P):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.act   = nn.GELU()
        self.do    = nn.Dropout2d(drop)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.do(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + x)
        return out

class CNNOnlyNet(nn.Module):
    """
    Pure 2D-CNN baseline:
    - Uses same conv/res blocks as your complex model.
    - No LSTM / BiLSTM / attention.
    - Predicts per-epoch logits independently.
    - Still accepts [B, T, C, H, W] for compatibility.
    """
    def __init__(self, n_classes=NUM_CLASSES, drop=DROPOUT_P):
        super().__init__()

        self.c1 = ConvBlock(3,   32, 3, 1, 1, drop)
        self.r1 = ResBlock(32, drop)
        self.c2 = ConvBlock(32, 64, 3, 2, 1, drop)
        self.r2 = ResBlock(64, drop)
        self.c3 = ConvBlock(64, 128, 3, 2, 1, drop)
        self.r3 = ResBlock(128, drop)
        self.c4 = ConvBlock(128, 128, 3, 1, 1, drop)
        self.r4 = ResBlock(128, drop)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = 128

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, train_mode=True):
        """
        x: [B, T, C, H, W]
        returns:
          main_logits: [B, T, n_classes]
          aux_logits:  [B, T, n_classes] (same as main, to reuse training loop)
        """
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = x.to(memory_format=torch.channels_last)

        h = self.c1(x); h = self.r1(h)
        h = self.c2(h); h = self.r2(h)
        h = self.c3(h); h = self.r3(h)
        h = self.c4(h); h = self.r4(h)

        h = self.pool(h).view(B * T, -1)     # [B*T, feat_dim]
        logits = self.classifier(h)          # [B*T, n_classes]

        logits = logits.view(B, T, -1)       # [B, T, n_classes]

        main_logits = logits
        aux_logits  = logits                 # so existing AUX_WEIGHT code works

        return main_logits, aux_logits

# -------------------- NORM & CLASS WEIGHTS --------------------
def zscore_from_loader(loader, device):
    m = None; s = None; n = 0
    for x, y, _ in loader:
        bs, t, c, h, w = x.shape
        x = x.view(bs * t, c, h, w).float().to(device, non_blocking=NON_BLOCKING)
        x_mean = x.mean(dim=(0, 2, 3))
        x_sq   = (x ** 2).mean(dim=(0, 2, 3))
        if m is None:
            m = x_mean * (bs * t)
            s = x_sq   * (bs * t)
        else:
            m += x_mean * (bs * t)
            s += x_sq   * (bs * t)
        n += (bs * t)
    mean = (m / max(n, 1))
    std  = (s / max(n, 1) - mean ** 2).clamp(min=1e-12).sqrt()
    return mean.detach(), std.detach()

def apply_norm(x, mean, std):
    # x: [B, T, C, H, W]
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

# -------------------- TRAIN / EVAL --------------------
def train_one_split(train_ds, val_ds, device, out_dir):
    pw = (NUM_WORKERS > 0)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, prefetch_factor=4 if pw else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, prefetch_factor=4 if pw else None
    )

    # z-score stats from training data only
    stats_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, prefetch_factor=4 if pw else None
    )
    mean, std = zscore_from_loader(stats_loader, device)
    mean, std = mean.to(device), std.to(device)

    def norm_batch(batch):
        x, y, sid = batch
        x = x.to(device, non_blocking=NON_BLOCKING).float()
        x = apply_norm(x, mean, std)
        y = y.to(device, non_blocking=NON_BLOCKING)
        return x, y, sid

    model = CNNOnlyNet().to(device)
    class_weights = get_class_weights([train_loader]).to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    scaler = amp.GradScaler(enabled=USE_AMP)

    best_val_f1 = -1.0
    best_epoch = None
    epochs_no_improve = 0

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{MAX_EPOCHS}"), 1):
            x, y, _ = norm_batch(batch)
            with amp.autocast(enabled=USE_AMP):
                main_logits, aux_logits = model(x, train_mode=True)
                loss_main = ce(main_logits.view(-1, NUM_CLASSES), y.view(-1))
                loss_aux  = ce(aux_logits.view(-1, NUM_CLASSES),  y.view(-1))
                loss = loss_main + AUX_WEIGHT * loss_aux
                if ACCUM_STEPS > 1:
                    loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step % ACCUM_STEPS) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            preds = main_logits.argmax(dim=-1)
            tr_correct += (preds.view(-1) == y.view(-1)).sum().item()
            tr_total   += y.numel()
            # approximate epoch loss normalized by #samples
            tr_loss    += loss.item() * x.size(0)

        train_loss = tr_loss / len(train_loader.dataset)
        train_acc  = tr_correct / tr_total

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"[GPU] max_allocated={torch.cuda.max_memory_allocated()/1e9:.2f} GB | "
                  f"reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # skip validation except at intervals (and last epoch)
        if epoch % VAL_EVERY != 0 and epoch != MAX_EPOCHS:
            continue

        # ---- Validation ----
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        all_y, all_p = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch}"):
                x, y, _ = norm_batch(batch)
                with amp.autocast(enabled=USE_AMP):
                    main_logits, aux_logits = model(x, train_mode=False)
                    loss_main = ce(main_logits.view(-1, NUM_CLASSES), y.view(-1))
                    loss_aux  = ce(aux_logits.view(-1, NUM_CLASSES),  y.view(-1))
                    loss = loss_main + AUX_WEIGHT * loss_aux

                va_loss += loss.item() * x.size(0)
                preds = main_logits.argmax(dim=-1)
                va_correct += (preds.view(-1) == y.view(-1)).sum().item()
                va_total   += y.numel()

                all_y.append(y.view(-1).cpu().numpy())
                all_p.append(preds.view(-1).cpu().numpy())

        val_loss = va_loss / len(val_loader.dataset)
        val_acc  = va_correct / va_total
        all_y = np.concatenate(all_y)
        all_p = np.concatenate(all_p)
        macro_f1 = f1_score(all_y, all_p, average="macro",
                            labels=list(range(NUM_CLASSES)), zero_division=0)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(macro_f1)

        improved = macro_f1 > best_val_f1 + 1e-6
        if improved:
            best_val_f1 = macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": mean.detach().cpu(),
                    "std": std.detach().cpu()
                },
                os.path.join(out_dir, "best.pt")
            )
        else:
            epochs_no_improve += 1

        print(f"[VAL] E{epoch} | tr_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | val MF1 {macro_f1:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"[EarlyStop] epoch {epoch}. Best MF1={best_val_f1:.4f} @ epoch {best_epoch}.")
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    if len(history["epoch"]) > 0:
        save_plot(history["epoch"], [history["train_loss"], history["val_loss"]],
                  ["Train Loss", "Val Loss"], "Loss", os.path.join(out_dir, "loss.png"))
        save_plot(history["epoch"], [history["train_acc"], history["val_acc"]],
                  ["Train Acc", "Val Acc"], "Accuracy", os.path.join(out_dir, "acc.png"))

    del model, opt, ce, scaler
    torch.cuda.empty_cache()

def evaluate_split(test_ds, device, out_dir, hasN3_subjects):
    pw = (NUM_WORKERS > 0)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, prefetch_factor=4 if pw else None
    )

    ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model = CNNOnlyNet().to(device)
    model.load_state_dict(ckpt["model"])
    mean = ckpt["mean"].to(device)
    std  = ckpt["std"].to(device)

    model.eval()

    def norm_batch(batch):
        x, y, sid = batch
        x = x.to(device, non_blocking=NON_BLOCKING).float()
        x = apply_norm(x, mean, std)
        y = y.to(device, non_blocking=NON_BLOCKING)
        return x, y, sid

    all_y, all_p, all_sid = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[Test]"):
            x, y, sid = norm_batch(batch)
            with amp.autocast(enabled=USE_AMP):
                main_logits, _ = model(x, train_mode=False)
                preds = main_logits.argmax(dim=-1)

            all_y.append(y.view(-1).cpu().numpy())
            all_p.append(preds.view(-1).cpu().numpy())
            all_sid += sid  # one sid per sequence

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)

    # Overall metrics
    acc   = accuracy_score(y, p)
    kappa = cohen_kappa_score(y, p, labels=list(range(NUM_CLASSES)))
    mf1   = f1_score(y, p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_f1 = f1_score(y, p, average=None,
                            labels=list(range(NUM_CLASSES)), zero_division=0)

    cm = confusion_matrix(y, p, labels=list(range(NUM_CLASSES)))
    plot_confmat(cm, [INV_LABELS[i] for i in range(NUM_CLASSES)],
                 "Confusion", os.path.join(out_dir, "confusion.png"))
    pd.DataFrame(
        cm,
        index=[f"T_{INV_LABELS[i]}" for i in range(NUM_CLASSES)],
        columns=[f"P_{INV_LABELS[i]}" for i in range(NUM_CLASSES)]
    ).to_csv(os.path.join(out_dir, "confusion.csv"))

    # Per-subject group: with N3 vs no N3
    per_epoch_sid = np.array([sid for sid in all_sid for _ in range(SEQ_LEN)])
    hasN3_set = set(hasN3_subjects)

    mask_withN3 = np.array([s in hasN3_set for s in per_epoch_sid])
    mask_noN3   = ~mask_withN3

    def subset_block(mask, tag):
        if mask.sum() == 0:
            return None
        y_sub = y[mask]
        p_sub = p[mask]
        cm_s  = confusion_matrix(y_sub, p_sub, labels=list(range(NUM_CLASSES)))
        plot_confmat(cm_s, [INV_LABELS[i] for i in range(NUM_CLASSES)],
                     f"Confusion ({tag})", os.path.join(out_dir, f"confusion_{tag}.png"))
        pd.DataFrame(
            cm_s,
            index=[f"T_{INV_LABELS[i]}" for i in range(NUM_CLASSES)],
            columns=[f"P_{INV_LABELS[i]}" for i in range(NUM_CLASSES)]
        ).to_csv(os.path.join(out_dir, f"confusion_{tag}.csv"))
        return {
            "acc": accuracy_score(y_sub, p_sub),
            "kappa": cohen_kappa_score(y_sub, p_sub, labels=list(range(NUM_CLASSES))),
            "macro_f1": f1_score(y_sub, p_sub, average="macro",
                                 labels=list(range(NUM_CLASSES)), zero_division=0)
        }

    withN3_metrics = subset_block(mask_withN3, "withN3")
    noN3_metrics   = subset_block(mask_noN3,   "noN3")

    summary = {
        "acc": float(acc),
        "kappa": float(kappa),
        "macro_f1": float(mf1),
        "per_class_f1": {INV_LABELS[i]: float(per_class_f1[i]) for i in range(NUM_CLASSES)},
        "withN3": withN3_metrics,
        "noN3": noN3_metrics
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    del model
    torch.cuda.empty_cache()

    return y, p, all_sid, summary

# -------------------- SUBJECT-LEVEL BUILDERS --------------------
def build_subject_index():
    pairs = list_all_subject_paths(DATA_ROOT)
    subj_items, subj_hasN3, subj_path = {}, {}, {}
    for sid, spath in tqdm(pairs, desc="Indexing subjects"):
        items = collect_subject_items(spath)
        if len(items) < SEQ_LEN:
            continue
        subj_items[sid] = items
        subj_path[sid]  = spath
        hasN3, _ = compute_has_flags(items)
        subj_hasN3[sid] = hasN3
    subjects = sorted(subj_items.keys(), key=natural_key)
    return subjects, subj_path, subj_items, subj_hasN3

def build_seqs_for_subjects(subj_list, subj_items):
    seqs, sids = [], []
    for s in subj_list:
        items = subj_items[s]
        seqs_s = make_all_sequences(items, SEQ_LEN, STRIDE_EPOCHS)
        seqs.extend(seqs_s)
        sids.extend([s] * len(seqs_s))
    return seqs, sids

# -------------------- MAIN --------------------
def main():
    print("Bismillah. Starting CNN-only baseline.")
    device = DEVICE
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] SEQ_LEN={SEQ_LEN}, STRIDE_EPOCHS={STRIDE_EPOCHS}, "
          f"BATCH_SIZE={BATCH_SIZE}, AMP={USE_AMP}")

    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA capability:", torch.cuda.get_device_capability(0))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    subjects, subj_path, subj_items, subj_hasN3 = build_subject_index()
    if len(subjects) < N_FOLDS:
        print(f"[ERROR] Need at least {N_FOLDS} subjects; found {len(subjects)}.")
        return

    print(f"[INFO] Subjects usable: {len(subjects)}")

    # Stratify based on presence of N3 in subject
    y_strat = np.array([subj_hasN3[s] for s in subjects], dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    overall_y_all, overall_p_all, fold_metrics = [], [], []
    start_fold = int(os.getenv("START_FOLD", "1"))
    end_fold   = int(os.getenv("END_FOLD",   str(N_FOLDS)))

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(subjects, y_strat), start=1):
        if not (start_fold <= fold_idx <= end_fold):
            continue

        fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n========== Fold {fold_idx}/{N_FOLDS} ==========")

        trainval_subjects = [subjects[i] for i in trainval_idx]
        test_subjects     = [subjects[i] for i in test_idx]

        # Inner split train/val
        y_trainval = np.array([subj_hasN3[s] for s in trainval_subjects], dtype=int)
        n_val = max(1, int(VAL_SUBJ_FRACTION * len(trainval_subjects)))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=n_val, random_state=SEED + fold_idx
        )
        train_idx_local, val_idx_local = next(
            splitter.split(np.zeros(len(trainval_subjects)), y_trainval)
        )
        train_subjects = [trainval_subjects[i] for i in train_idx_local]
        val_subjects   = [trainval_subjects[i] for i in val_idx_local]

        # Build sequences
        train_seqs, train_sids = build_seqs_for_subjects(train_subjects, subj_items)
        val_seqs,   val_sids   = build_seqs_for_subjects(val_subjects,   subj_items)
        test_seqs,  test_sids  = build_seqs_for_subjects(test_subjects,  subj_items)

        # Leakage checks
        assert_disjoint(train_seqs, val_seqs,  "train", "val")
        assert_disjoint(train_seqs, test_seqs, "train", "test")
        assert_disjoint(val_seqs,   test_seqs, "val",   "test")

        # Datasets
        train_ds = ModSpecSeqDataset(train_seqs, train_sids)
        val_ds   = ModSpecSeqDataset(val_seqs,   val_sids)
        test_ds  = ModSpecSeqDataset(test_seqs,  test_sids)

        # Train / Evaluate
        train_one_split(train_ds, val_ds, device, fold_dir)

        test_hasN3_subjects = [s for s in test_subjects if subj_hasN3.get(s, 0) == 1]
        y, p, sid, summary = evaluate_split(test_ds, device, fold_dir,
                                            hasN3_subjects=test_hasN3_subjects)

        fold_metrics.append({
            "fold": fold_idx,
            "accuracy": summary["acc"],
            "kappa": summary["kappa"],
            "macro_f1": summary["macro_f1"]
        })

        overall_y_all.append(y)
        overall_p_all.append(p)

        del train_ds, val_ds, test_ds
        torch.cuda.empty_cache()

    # ----- OVERALL (across folds) -----
    overall_y = np.concatenate(overall_y_all)
    overall_p = np.concatenate(overall_p_all)

    cm = confusion_matrix(overall_y, overall_p, labels=list(range(NUM_CLASSES)))
    plot_confmat(cm, [INV_LABELS[i] for i in range(NUM_CLASSES)],
                 "Overall Confusion (5-fold)", os.path.join(RESULTS_DIR, "overall_confusion.png"))
    pd.DataFrame(
        cm,
        index=[f"T_{INV_LABELS[i]}" for i in range(NUM_CLASSES)],
        columns=[f"P_{INV_LABELS[i]}" for i in range(NUM_CLASSES)]
    ).to_csv(os.path.join(RESULTS_DIR, "overall_confusion.csv"))

    acc   = accuracy_score(overall_y, overall_p)
    kappa = cohen_kappa_score(overall_y, overall_p, labels=list(range(NUM_CLASSES)))
    mf1   = f1_score(overall_y, overall_p, average="macro",
                     labels=list(range(NUM_CLASSES)), zero_division=0)

    df_folds = pd.DataFrame(fold_metrics)
    df_folds.to_csv(os.path.join(RESULTS_DIR, "fold_metrics.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "overall_metrics.json"), "w") as f:
        json.dump({
            "overall_accuracy": float(acc),
            "overall_kappa": float(kappa),
            "overall_macro_f1": float(mf1),
            "folds": fold_metrics,
            "mean_accuracy": float(df_folds["accuracy"].mean()),
            "std_accuracy":  float(df_folds["accuracy"].std(ddof=0)),
            "mean_kappa":    float(df_folds["kappa"].mean()),
            "std_kappa":     float(df_folds["kappa"].std(ddof=0)),
            "mean_macro_f1": float(df_folds["macro_f1"].mean()),
            "std_macro_f1":  float(df_folds["macro_f1"].std(ddof=0))
        }, f, indent=2)

    print("\n===== OVERALL (5-fold, CNN-only) =====")
    print(f"Acc={acc:.4f}  Kappa={kappa:.4f}  MacroF1={mf1:.4f}")
    print("Per-fold (mean±std): "
          f"Acc={df_folds['accuracy'].mean():.4f}±{df_folds['accuracy'].std(ddof=0):.4f}, "
          f"Kappa={df_folds['kappa'].mean():.4f}±{df_folds['kappa'].std(ddof=0):.4f}, "
          f"MacroF1={df_folds['macro_f1'].mean():.4f}±{df_folds['macro_f1'].std(ddof=0):.4f}")

if __name__ == "__main__":
    main()
