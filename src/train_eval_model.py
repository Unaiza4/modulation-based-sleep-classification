# src/train_eval_modulation.py

import os, json, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch import amp
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from modulation_core import (
    DATA_ROOT, SEQ_LEN, STRIDE_EPOCHS, VAL_SUBJ_FRACTION,
    CLASS_LABELS, INV_LABELS, NUM_CLASSES,
    ModSpecSeqDataset, EEGSNetLike,
    zscore_from_loader, apply_norm, get_class_weights,
    list_all_subject_paths, collect_subject_items,
    make_all_sequences, compute_has_flags,
    assert_disjoint, save_plot, plot_confmat, natural_key
)

# -------------------- DEVICE / TRAINING CONFIG --------------------
RESULTS_DIR = ""   # your input directory

MAX_EPOCHS   = 150
BATCH_SIZE   = 2
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
PATIENCE     = 10
VAL_EVERY    = 3
AUX_WEIGHT   = 0.5
ACCUM_STEPS  = 1
N_FOLDS      = 5
SEED         = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

cudnn.benchmark = True
cudnn.deterministic = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY   = (DEVICE.type == "cuda")
NON_BLOCKING = (DEVICE.type == "cuda")
USE_AMP      = (DEVICE.type == "cuda")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# -------------------- TRAIN / EVAL FUNCTIONS --------------------
def train_one_split(train_ds, val_ds, device, out_dir):
    pw = (NUM_WORKERS > 0)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, prefetch_factor=4 if pw else None
    )
    val_loader   = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, prefetch_factor=4 if pw else None
    )

    # z-score from training only
    stats_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, prefetch_factor=4 if pw else None
    )
    mean, std = zscore_from_loader(stats_loader, device)
    mean, std = mean.to(device), std.to(device)

    def norm_batch(batch):
        x,y,sid = batch
        # keep standard layout here; channels_last is applied inside model on 4D tensors
        x = x.to(device, non_blocking=NON_BLOCKING).float()
        x = apply_norm(x, mean, std)
        return x, y.to(device, non_blocking=NON_BLOCKING), sid

    model = EEGSNetLike().to(device)
    class_weights = get_class_weights([train_loader]).to(device)
    ce  = nn.CrossEntropyLoss(weight=class_weights)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    # defaults
    best_val_f1 = -1.0
    best_epoch = None
    epochs_no_improve = 0
    history = {"epoch":[], "train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    start_epoch = 1

    # --------- RESUME LOGIC ---------
    if os.path.isfile(ckpt_path):
        print(f"[Resume] Loading checkpoint from {ckpt_path}")
        ckpt_state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt_state["model"])
        opt.load_state_dict(ckpt_state["opt"])
        scheduler.load_state_dict(ckpt_state["scheduler"])
        scaler.load_state_dict(ckpt_state["scaler"])
        mean = ckpt_state["mean"].to(device)
        std  = ckpt_state["std"].to(device)
        best_val_f1 = ckpt_state.get("best_val_f1", best_val_f1)
        best_epoch  = ckpt_state.get("best_epoch", best_epoch)
        epochs_no_improve = ckpt_state.get("epochs_no_improve", epochs_no_improve)
        history = ckpt_state.get("history", history)
        start_epoch = ckpt_state.get("epoch", 0) + 1
        finished = ckpt_state.get("finished", False)

        if finished or start_epoch > MAX_EPOCHS:
            print("[Resume] Training for this fold already finished. Skipping training.")
            return

        print(f"[Resume] Continuing from epoch {start_epoch} (best MF1={best_val_f1:.4f})")

    # --------- TRAINING LOOP ---------
    for epoch in range(start_epoch, MAX_EPOCHS+1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{MAX_EPOCHS}"), 1):
            x,y,_ = norm_batch(batch)
            with amp.autocast('cuda', enabled=USE_AMP):
                main_logits, aux_logits = model(x, train_mode=True)
                loss_main = ce(main_logits.view(-1, NUM_CLASSES), y.view(-1))
                loss_aux  = ce(aux_logits.view(-1, NUM_CLASSES),  y.view(-1))
                loss = loss_main + AUX_WEIGHT * loss_aux
                if ACCUM_STEPS > 1:
                    loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step % ACCUM_STEPS == 0):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            preds = main_logits.argmax(dim=-1)
            tr_correct += (preds.view(-1) == y.view(-1)).sum().item()
            tr_total   += y.numel()
            tr_loss    += loss.item() * (x.size(0) if ACCUM_STEPS == 1 else x.size(0) * ACCUM_STEPS)

        train_loss = tr_loss / len(train_loader.dataset)
        train_acc  = tr_correct / tr_total

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"[GPU] max_allocated={torch.cuda.max_memory_allocated()/1e9:.2f} GB | "
                  f"reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # Only validate every VAL_EVERY epochs (same as before)
        if epoch % VAL_EVERY != 0 and epoch != MAX_EPOCHS:
            # still save running checkpoint so we can resume mid-fold
            ckpt_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "mean": mean.detach().cpu(),
                "std": std.detach().cpu(),
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
                "history": history,
                "finished": False,
            }
            torch.save(ckpt_state, ckpt_path)
            continue

        # ---- Validation ----
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        all_y, all_p = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch}"):
                x,y,_ = norm_batch(batch)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
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
        all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
        macro_f1 = f1_score(all_y, all_p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)

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
            torch.save({"model":model.state_dict(),
                        "mean":mean.detach().cpu(),
                        "std":std.detach().cpu()}, os.path.join(out_dir,"best.pt"))
        else:
            epochs_no_improve += 1

        print(f"[VAL] E{epoch} | tr_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | val MF1 {macro_f1:.4f}")

        # ---- SAVE CHECKPOINT (for resume) ----
        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "history": history,
            "finished": False,
        }
        torch.save(ckpt_state, ckpt_path)

        if epochs_no_improve >= PATIENCE:
            print(f"[EarlyStop] epoch {epoch}. Best MF1={best_val_f1:.4f} @ epoch {best_epoch}.")
            break

    # mark training finished in checkpoint
    if os.path.isfile(ckpt_path):
        ckpt_state = torch.load(ckpt_path, map_location="cpu")
        ckpt_state["finished"] = True
        torch.save(ckpt_state, ckpt_path)

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    if len(history["epoch"]) > 0:
        save_plot(history["epoch"], [history["train_loss"], history["val_loss"]],
                  ["Train Loss","Val Loss"], "Loss", os.path.join(out_dir,"loss.png"))
        save_plot(history["epoch"], [history["train_acc"], history["val_acc"]],
                  ["Train Acc","Val Acc"], "Accuracy", os.path.join(out_dir,"acc.png"))

    # free some GPU RAM before evaluation/next fold
    del model, opt, ce, scaler
    torch.cuda.empty_cache()

def evaluate_split(test_ds, device, out_dir, hasN3_subjects):
    pw = (NUM_WORKERS > 0)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, prefetch_factor=4 if pw else None
    )
    ckpt = torch.load(os.path.join(out_dir,"best.pt"), map_location=device)
    model = EEGSNetLike().to(device)
    model.load_state_dict(ckpt["model"])
    mean = ckpt["mean"].to(device); std = ckpt["std"].to(device)
    model.eval()

    def norm_batch(batch):
        x,y,sid = batch
        x = x.to(device, non_blocking=NON_BLOCKING).float()
        x = apply_norm(x, mean, std)
        return x, y.to(device, non_blocking=NON_BLOCKING), sid

    all_y, all_p, all_sid = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[Test]"):
            x,y,sid = norm_batch(batch)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                main_logits, _ = model(x, train_mode=False)
                preds = main_logits.argmax(dim=-1)
            all_y.append(y.view(-1).cpu().numpy())
            all_p.append(preds.view(-1).cpu().numpy())
            all_sid += sid

    y = np.concatenate(all_y); p = np.concatenate(all_p)

    # Overall
    acc  = accuracy_score(y, p)
    kappa= cohen_kappa_score(y, p, labels=list(range(NUM_CLASSES)))
    mf1  = f1_score(y, p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_f1 = f1_score(y, p, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    cm = confusion_matrix(y, p, labels=list(range(NUM_CLASSES)))
    plot_confmat(cm, [INV_LABELS[i] for i in range(NUM_CLASSES)],
                 "Confusion", os.path.join(out_dir, "confusion.png"))
    pd.DataFrame(cm, index=[f"T_{INV_LABELS[i]}" for i in range(NUM_CLASSES)],
                    columns=[f"P_{INV_LABELS[i]}" for i in range(NUM_CLASSES)]).to_csv(
                        os.path.join(out_dir,"confusion.csv"))

    # withN3 / noN3 per subject
    per_epoch_sid = np.array([sid for sid in all_sid for _ in range(SEQ_LEN)])
    hasN3_set = set(hasN3_subjects)

    mask_withN3 = np.array([s in hasN3_set for s in per_epoch_sid])
    mask_noN3   = ~mask_withN3

    def subset_block(mask, tag):
        if mask.sum() == 0:
            return None
        y_sub = y[mask]; p_sub = p[mask]
        cm_s  = confusion_matrix(y_sub, p_sub, labels=list(range(NUM_CLASSES)))
        plot_confmat(cm_s, [INV_LABELS[i] for i in range(NUM_CLASSES)],
                     f"Confusion ({tag})", os.path.join(out_dir, f"confusion_{tag}.png"))
        pd.DataFrame(cm_s, index=[f"T_{INV_LABELS[i]}" for i in range(NUM_CLASSES)],
                        columns=[f"P_{INV_LABELS[i]}" for i in range(NUM_CLASSES)]).to_csv(
                            os.path.join(out_dir, f"confusion_{tag}.csv"))
        return {
            "acc": accuracy_score(y_sub, p_sub),
            "kappa": cohen_kappa_score(y_sub, p_sub, labels=list(range(NUM_CLASSES))),
            "macro_f1": f1_score(y_sub, p_sub, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
        }

    withN3_metrics = subset_block(mask_withN3, "withN3")
    noN3_metrics   = subset_block(mask_noN3,   "noN3")

    summary = {
        "acc": float(acc), "kappa": float(kappa), "macro_f1": float(mf1),
        "per_class_f1": {INV_LABELS[i]: float(per_class_f1[i]) for i in range(NUM_CLASSES)},
        "withN3": withN3_metrics, "noN3": noN3_metrics
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # free some GPU RAM
    del model
    torch.cuda.empty_cache()

    return y, p, all_sid, summary

# -------------------- SUBJECT HELPERS --------------------
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
        seqs_s = make_all_sequences(items, seq_len=SEQ_LEN, stride=STRIDE_EPOCHS)
        seqs.extend(seqs_s)
        sids.extend([s] * len(seqs_s))
    return seqs, sids

# -------------------- MAIN CROSS-VALIDATION --------------------
def main():
    device = DEVICE
    print(f"[INFO] Using device: {device}")
    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA capability:", torch.cuda.get_device_capability(0))

    subjects, subj_path, subj_items, subj_hasN3 = build_subject_index()
    if len(subjects) < N_FOLDS:
        print(f"[ERROR] Need at least {N_FOLDS} subjects; found {len(subjects)}.")
        return
    print(f"[INFO] Subjects usable: {len(subjects)}")

    y_strat = np.array([subj_hasN3[s] for s in subjects], dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    overall_y_all, overall_p_all, fold_metrics = [], [], []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(subjects, y_strat), start=1):
        fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n========== Fold {fold_idx}/{N_FOLDS} ==========")

        trainval_subjects = [subjects[i] for i in trainval_idx]
        test_subjects     = [subjects[i] for i in test_idx]

        y_trainval = np.array([subj_hasN3[s] for s in trainval_subjects], dtype=int)
        n_val = max(1, int(VAL_SUBJ_FRACTION * len(trainval_subjects)))
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=SEED + fold_idx)
        train_idx_local, val_idx_local = next(splitter.split(np.zeros(len(trainval_subjects)), y_trainval))
        train_subjects = [trainval_subjects[i] for i in train_idx_local]
        val_subjects   = [trainval_subjects[i] for i in val_idx_local]

        train_seqs, train_sids = build_seqs_for_subjects(train_subjects, subj_items)
        val_seqs,   val_sids   = build_seqs_for_subjects(val_subjects,   subj_items)
        test_seqs,  test_sids  = build_seqs_for_subjects(test_subjects,  subj_items)

        assert_disjoint(train_seqs, val_seqs,  "train", "val")
        assert_disjoint(train_seqs, test_seqs, "train", "test")
        assert_disjoint(val_seqs,   test_seqs, "val",   "test")

        train_ds = ModSpecSeqDataset(train_seqs, train_sids)
        val_ds   = ModSpecSeqDataset(val_seqs,   val_sids)
        test_ds  = ModSpecSeqDataset(test_seqs,  test_sids)

        train_one_split(train_ds, val_ds, device, fold_dir)

        test_hasN3_subjects = [s for s in test_subjects if subj_hasN3.get(s,0)==1]
        y, p, sid, summary = evaluate_split(test_ds, device, fold_dir, hasN3_subjects=test_hasN3_subjects)

        fold_metrics.append({
            "fold": fold_idx,
            "accuracy": summary["acc"],
            "kappa": summary["kappa"],
            "macro_f1": summary["macro_f1"]
        })
        overall_y_all.append(y); overall_p_all.append(p)

        del train_ds, val_ds, test_ds
        torch.cuda.empty_cache()

    if len(overall_y_all) == 0:
        print("[WARN] No folds were run. Nothing to aggregate.")
        return

    overall_y = np.concatenate(overall_y_all); overall_p = np.concatenate(overall_p_all)

    cm = confusion_matrix(overall_y, overall_p, labels=list(range(NUM_CLASSES)))
    plot_confmat(cm, [INV_LABELS[i] for i in range(NUM_CLASSES)],
                 "Overall Confusion (5-fold)", os.path.join(RESULTS_DIR,"overall_confusion.png"))
    pd.DataFrame(cm,
                 index=[f"T_{INV_LABELS[i]}" for i in range(NUM_CLASSES)],
                 columns=[f"P_{INV_LABELS[i]}" for i in range(NUM_CLASSES)]
    ).to_csv(os.path.join(RESULTS_DIR,"overall_confusion.csv"))

    acc  = accuracy_score(overall_y, overall_p)
    kappa= cohen_kappa_score(overall_y, overall_p, labels=list(range(NUM_CLASSES)))
    mf1  = f1_score(overall_y, overall_p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)

    df_folds = pd.DataFrame(fold_metrics)
    df_folds.to_csv(os.path.join(RESULTS_DIR, "fold_metrics.csv"), index=False)

    with open(os.path.join(RESULTS_DIR,"overall_metrics.json"), "w") as f:
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

    print("\n===== OVERALL (5-fold) =====")
    print(f"Acc={acc:.4f}  Kappa={kappa:.4f}  MacroF1={mf1:.4f}")


if __name__ == "__main__":
    main()
