#!/usr/bin/env python3
# ==========================
# GPU NN COMPLETENESS SWEEP (robust drop frame)
# ==========================

import argparse
import itertools
import random
from math import cos, pi
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from nn_utils import plot_lr_wd_grid, plot_train_val_loss, scatter_param_vs_val

# ---- CONFIG ---
SEED = 42
BATCH_SIZE = 4096
EPOCHS_SWEEP = 80
EPOCHS_BEST = 110
WARMUP_FRAC = 0.1

HIDDEN_DIMS = [512]
N_HIDDEN_LIST = [2]
WEIGHT_DECAYS = [1e-3, 5e-3, 1e-2, 3e-2]
MAX_LRS = [5e-3, 1e-2, 5e-2]

# ---- DROP FRAME CONFIG (default values; CLI overrides) ---
DROP_FRAME = 0
CLUSTERS_PER_FRAME = 500
N_FRAMES_DEFAULT = 50
N_REFF_DEFAULT = 10


# ==========================
# UTILS
# ==========================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_flat_frame_ids(
    *,
    n_clusters: int,
    n_frames: int,
    n_reff: int,
    order: str,
) -> np.ndarray:
    """
    Build frame_id for each flattened sample.

    order:
      - 'CFR' : flatten order is cluster -> frame -> reff (your current allprop_18Dec_final.npz.npy)
               index = c*(n_frames*n_reff) + f*n_reff + r
      - 'FRC' : frame -> reff -> cluster
      - 'FCR' : frame -> cluster -> reff
      - 'CRF' : cluster -> reff -> frame
    """
    if order not in {"CFR", "FRC", "FCR", "CRF"}:
        raise ValueError(f"Unsupported flatten order: {order}")

    if order == "CFR":
        # c slowest, r fastest inside f
        # [c0: f0 r0..][c0: f1 r0..]..[c1: f0 r0..]...
        return np.repeat(np.tile(np.arange(n_frames), n_reff), n_clusters)
    if order == "FRC":
        # f slowest, c fastest inside r
        # for f: for r: for c
        return np.repeat(np.arange(n_frames), n_reff * n_clusters)
    if order == "FCR":
        # for f: for c: for r
        return np.repeat(np.arange(n_frames), n_clusters * n_reff)
    # order == "CRF": for c: for r: for f
    return np.tile(np.arange(n_frames), n_clusters * n_reff)


def drop_frame_from_prop_flat(
    prop_raw: dict,
    *,
    drop_frame: int,
    n_clusters: int,
    n_frames: int,
    n_reff: int,
    flatten_order: str,
) -> tuple[dict, np.ndarray]:
    """
    Drop a frame from flattened prop arrays (axis0 length = n_clusters*n_frames*n_reff).
    Returns: (prop_clean, keep_mask)
    """
    for k in ("mass", "age", "av", "phot"):
        if k not in prop_raw:
            raise KeyError(f"prop missing required key: {k}")

    n0 = prop_raw["mass"].shape[0]
    expected = n_clusters * n_frames * n_reff
    assert (
        n0 == expected
    ), f"prop['mass'] length mismatch: got {n0}, expected {expected}={n_clusters}*{n_frames}*{n_reff}"

    if not (0 <= drop_frame < n_frames):
        raise AssertionError(f"drop_frame={drop_frame} out of range [0, {n_frames-1}]")

    frame_ids = build_flat_frame_ids(
        n_clusters=n_clusters, n_frames=n_frames, n_reff=n_reff, order=flatten_order
    )
    assert frame_ids.shape[0] == n0, "frame_id length mismatch"

    # should remove exactly n_clusters*n_reff samples
    n_to_drop = int(np.sum(frame_ids == drop_frame))
    assert (
        n_to_drop == n_clusters * n_reff
    ), f"Expected to drop {n_clusters*n_reff} samples, but frame_ids marks {n_to_drop}"

    keep = frame_ids != drop_frame
    assert keep.shape[0] == n0
    assert int(np.sum(~keep)) == n_clusters * n_reff

    prop = {}
    for k, v in prop_raw.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n0:
            prop[k] = v[keep]
        else:
            prop[k] = v

    # post-assertions
    n1 = prop["mass"].shape[0]
    assert n1 == n0 - n_clusters * n_reff, f"After drop, expected {n0 - n_clusters*n_reff}, got {n1}"

    # sanity: no dropped frame remains
    frame_ids_after = frame_ids[keep]
    assert not np.any(frame_ids_after == drop_frame), "Dropped frame still present after masking"

    return prop, keep


def drop_frame_from_det_3d(
    det_raw: np.ndarray,
    *,
    drop_frame: int,
) -> np.ndarray:
    """
    det_raw is expected to be (n_clusters, n_frames, n_reff).
    Drop along axis=1 (frame axis).
    """
    assert det_raw.ndim == 3, f"det_raw expected 3D, got shape {det_raw.shape}"
    n_clusters, n_frames, n_reff = det_raw.shape
    assert 0 <= drop_frame < n_frames, f"drop_frame={drop_frame} out of range [0, {n_frames-1}]"
    det = np.delete(det_raw, drop_frame, axis=1)
    assert det.shape == (n_clusters, n_frames - 1, n_reff)
    return det


def flatten_det_like_training(det_3d: np.ndarray, *, flatten_order: str) -> np.ndarray:
    """
    Produce 1D y vector aligned with prop flatten order.
    Your original training used:
        y = det.astype(np.float32).reshape(-1, 1)
    which corresponds to NumPy C-order flatten on det array layout.

    But your prop flatten in allprop_18Dec_final.npz.npy is CFR (cluster->frame->reff).
    det_raw layout is (cluster, frame, reff). Flattening det in C-order is CFR already.
    So for CFR, det_3d.reshape(-1) is aligned.

    For other orders, you must transpose before reshape.
    """
    if flatten_order == "CFR":
        return det_3d.reshape(-1)
    if flatten_order == "FRC":
        return det_3d.transpose(1, 2, 0).reshape(-1)
    if flatten_order == "FCR":
        return det_3d.transpose(1, 0, 2).reshape(-1)
    if flatten_order == "CRF":
        return det_3d.transpose(0, 2, 1).reshape(-1)
    raise ValueError(f"Unsupported flatten order: {flatten_order}")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_lr_schedule(total_steps: int, max_lr: float, warmup_frac: float = 0.1):
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_fn(step: int) -> float:
        step = min(step, total_steps - 1)
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * max_lr * (1 + cos(pi * progress))

    return lr_fn


def train_one_model(
    x_tr_t,
    y_tr_t,
    x_va_t,
    y_va_t,
    input_dim,
    hidden_dim,
    n_hidden,
    max_lr,
    weight_decay,
    epochs,
    batch_size,
    warmup_frac,
    run_name,
    device,
    pos_weight,
    save_best=False,
    save_path=None,
):
    model = MLP(input_dim, hidden_dim, n_hidden).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = x_tr_t.shape[0]
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    lr_schedule = make_lr_schedule(total_steps, max_lr, warmup_frac)

    history = {
        "train_loss_epoch": [],
        "val_loss_epoch": [],
        "config": {
            "hidden_dim": hidden_dim,
            "n_hidden": n_hidden,
            "max_lr": max_lr,
            "weight_decay": weight_decay,
        },
    }

    best_val_loss = float("inf")
    global_step = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        train_loss_sum = 0.0
        assert x_tr_t.dtype == torch.float32
        assert y_tr_t.dtype == torch.float32

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_tr_t[idx], y_tr_t[idx]

            lr = lr_schedule(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            global_step += 1

        train_loss = train_loss_sum / n

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_va_t), y_va_t).item()

        history["train_loss_epoch"].append(train_loss)
        history["val_loss_epoch"].append(val_loss)

        if save_best and save_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "n_hidden": n_hidden,
                    },
                    "best_val_loss": best_val_loss,
                    "pos_weight": float(pos_weight.item()),
                },
                save_path,
            )

        print(
            f"[{run_name}] Epoch {ep + 1:02d}/{epochs} "
            f"Train CE={train_loss:.4f}  Val CE={val_loss:.4f}"
        )

    return model, history


def run_sweep_for_feature_set(
    name,
    x_tr_t,
    y_tr_t,
    x_va_t,
    y_va_t,
    device,
    pos_weight,
):
    input_dim = x_tr_t.shape[1]
    results = []

    for h, n_hidden, wd, lr in itertools.product(
        HIDDEN_DIMS, N_HIDDEN_LIST, WEIGHT_DECAYS, MAX_LRS
    ):
        run_name = f"{name}_h{h}_L{n_hidden}_wd{wd:.0e}_lr{lr:.0e}"
        _, hist = train_one_model(
            x_tr_t,
            y_tr_t,
            x_va_t,
            y_va_t,
            input_dim,
            h,
            n_hidden,
            lr,
            wd,
            EPOCHS_SWEEP,
            BATCH_SIZE,
            WARMUP_FRAC,
            run_name,
            device,
            pos_weight,
        )
        cfg = hist["config"].copy()
        cfg["final_val_loss"] = hist["val_loss_epoch"][-1]
        results.append((cfg, hist))

    results.sort(key=lambda x: x[0]["final_val_loss"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-path", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--out-dir", default="./nn_sweep_out")
    parser.add_argument("--outname", default="model0")
    parser.add_argument("--save-best", action="store_true")

    # robust drop controls
    parser.add_argument("--drop-frame", type=int, default=DROP_FRAME)
    parser.add_argument("--clusters-per-frame", type=int, default=CLUSTERS_PER_FRAME)
    parser.add_argument("--nframes", type=int, default=N_FRAMES_DEFAULT)
    parser.add_argument("--nreff", type=int, default=N_REFF_DEFAULT)

    # IMPORTANT: your current allprop flatten is CFR (cluster->frame->reff)
    parser.add_argument("--prop-flatten-order", type=str, default="CFR")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dir = out_dir / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- load ----
    prop_raw = np.load(args.npz_path, allow_pickle=True).item()
    det_raw = np.load(args.det_path)

    # ---- assert raw det expected 3D: (clusters, frames, reff) ----
    assert det_raw.ndim == 3, f"det expected shape (clusters, frames, reff), got {det_raw.shape}"
    det_ncl, det_nf, det_nr = det_raw.shape
    assert det_ncl == args.clusters_per_frame, (
        f"det clusters axis mismatch: det has {det_ncl}, "
        f"clusters_per_frame={args.clusters_per_frame}"
    )
    assert det_nf == args.nframes, f"det frames mismatch: det has {det_nf}, nframes={args.nframes}"
    assert det_nr == args.nreff, f"det reff mismatch: det has {det_nr}, nreff={args.nreff}"

    # ---- assert prop lengths ----
    for k in ("mass", "age", "av", "phot"):
        assert k in prop_raw, f"prop missing key: {k}"

    n0 = prop_raw["mass"].shape[0]
    expected_n0 = args.clusters_per_frame * args.nframes * args.nreff
    assert n0 == expected_n0, (
        f"prop['mass'] length mismatch: got {n0}, expected {expected_n0}="
        f"{args.clusters_per_frame}*{args.nframes}*{args.nreff}"
    )

    # also ensure key arrays align
    for k in ("age", "av"):
        assert (
            prop_raw[k].shape[0] == n0
        ), f"prop[{k}] length mismatch: got {prop_raw[k].shape[0]}, expected {n0}"
    assert (
        prop_raw["phot"].shape[0] == n0
    ), f"prop['phot'] length mismatch: got {prop_raw['phot'].shape[0]}, expected {n0}"

    # ---- DROP FRAME robustly ----
    # 1) drop from det in its native 3D structure
    det_3d = drop_frame_from_det_3d(det_raw, drop_frame=args.drop_frame)
    assert det_3d.shape[1] == args.nframes - 1

    # 2) drop from prop using explicit frame_ids that match the prop flatten order
    prop, keep_mask = drop_frame_from_prop_flat(
        prop_raw,
        drop_frame=args.drop_frame,
        n_clusters=args.clusters_per_frame,
        n_frames=args.nframes,
        n_reff=args.nreff,
        flatten_order=args.prop_flatten_order,
    )

    # 3) flatten det to match prop order (CFR by default)
    y_flat = flatten_det_like_training(det_3d, flatten_order=args.prop_flatten_order).astype(
        np.float32
    )
    assert y_flat.shape[0] == prop["mass"].shape[0], (
        f"After dropping, y length {y_flat.shape[0]} != prop length {prop['mass'].shape[0]}"
    )

    print(
        f"✔ Dropped frame={args.drop_frame}. "
        f"prop N: {n0} -> {prop['mass'].shape[0]} (dropped {args.clusters_per_frame*args.nreff}). "
        f"det: {det_raw.shape} -> {det_3d.shape}."
    )

    # ---- build features/labels ----
    x_phys = np.column_stack([prop["mass"], prop["age"], prop["av"]])
    x_phot = prop["phot"]
    y = y_flat.reshape(-1, 1)

    # ---- split ----
    y_int = y_flat.astype(int)
    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(sss.split(x_phys, y_int))

    np.savez(out_dir / "split_indices.npz", train_idx=tr_idx, val_idx=te_idx)

    # ---- scalers (train only) ----
    scaler_phys = StandardScaler().fit(x_phys[tr_idx])
    scaler_phot = StandardScaler().fit(x_phot[tr_idx])
    joblib.dump(scaler_phys, out_dir / f"scaler_phys_{args.outname}.pkl")
    joblib.dump(scaler_phot, out_dir / f"scaler_phot_{args.outname}.pkl")

    xp_tr = torch.tensor(scaler_phys.transform(x_phys[tr_idx]), dtype=torch.float32, device=device)
    xp_va = torch.tensor(scaler_phys.transform(x_phys[te_idx]), dtype=torch.float32, device=device)
    xph_tr = torch.tensor(scaler_phot.transform(x_phot[tr_idx]), dtype=torch.float32, device=device)
    xph_va = torch.tensor(scaler_phot.transform(x_phot[te_idx]), dtype=torch.float32, device=device)

    y_tr = torch.tensor(y[tr_idx], dtype=torch.float32, device=device)
    y_va = torch.tensor(y[te_idx], dtype=torch.float32, device=device)

    pos_frac = float(y_tr.mean())
    pos_weight = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)], device=device)

    # ---- sweep + train best (phys) ----
    results_phys = run_sweep_for_feature_set("phys", xp_tr, y_tr, xp_va, y_va, device, pos_weight)
    best_phys = results_phys[0][0]

    train_one_model(
        xp_tr,
        y_tr,
        xp_va,
        y_va,
        input_dim=xp_tr.shape[1],
        hidden_dim=best_phys["hidden_dim"],
        n_hidden=best_phys["n_hidden"],
        max_lr=best_phys["max_lr"],
        weight_decay=best_phys["weight_decay"],
        epochs=EPOCHS_BEST,
        batch_size=BATCH_SIZE,
        warmup_frac=WARMUP_FRAC,
        run_name="phys_best",
        device=device,
        pos_weight=pos_weight,
        save_best=args.save_best,
        save_path=save_dir / f"best_model_phys_{args.outname}.pt",
    )

    best_cfg, best_hist = results_phys[0]
    plot_train_val_loss(
        best_hist,
        title=(
            f"Train vs Val loss (phys-best)\n"
            f"h={best_cfg['hidden_dim']}, L={best_cfg['n_hidden']}, "
            f"lr={best_cfg['max_lr']:.1e}, wd={best_cfg['weight_decay']:.1e}"
        ),
        outpath=out_dir / f"phys_best_train_val_{args.outname}.png",
    )
    scatter_param_vs_val(
        results_phys,
        x_key="max_lr",
        title="Validation loss vs max LR (phys)",
        outpath=out_dir / f"phys_val_vs_lr_{args.outname}.png",
    )
    scatter_param_vs_val(
        results_phys,
        x_key="weight_decay",
        title="Validation loss vs weight decay (phys)",
        outpath=out_dir / f"phys_val_vs_wd_{args.outname}.png",
    )

    plot_lr_wd_grid(
    results_phys,
    title="Validation loss landscape (phys)",
    outpath=out_dir / f"phys_lr_wd_grid_{args.outname}.png",
    save_values_path=out_dir / f"phys_lr_wd_grid_{args.outname}.npz",)

    # ---- sweep + train best (phot) ----
    results_phot = run_sweep_for_feature_set("phot", xph_tr, y_tr, xph_va, y_va, device, pos_weight)
    best_phot = results_phot[0][0]

    best_cfg, best_hist = results_phot[0]
    plot_train_val_loss(
        best_hist,
        title=(
            f"Train vs Val loss (phot-best)\n"
            f"h={best_cfg['hidden_dim']}, L={best_cfg['n_hidden']}, "
            f"lr={best_cfg['max_lr']:.1e}, wd={best_cfg['weight_decay']:.1e}"
        ),
        outpath=out_dir / f"phot_best_train_val_{args.outname}.png",
    )
    scatter_param_vs_val(
        results_phot,
        x_key="max_lr",
        title="Validation loss vs max LR (phot)",
        outpath=out_dir / f"phot_val_vs_lr_{args.outname}.png",
    )
    scatter_param_vs_val(
        results_phot,
        x_key="weight_decay",
        title="Validation loss vs weight decay (phot)",
        outpath=out_dir / f"phot_val_vs_wd_{args.outname}.png",
    )
    plot_lr_wd_grid(
    results_phot,
    title="Validation loss landscape (phot)",
    outpath=out_dir / f"phot_lr_wd_grid_{args.outname}.png",
    save_values_path=out_dir / f"phot_lr_wd_grid_{args.outname}.npz",)

    train_one_model(
        xph_tr,
        y_tr,
        xph_va,
        y_va,
        input_dim=xph_tr.shape[1],
        hidden_dim=best_phot["hidden_dim"],
        n_hidden=best_phot["n_hidden"],
        max_lr=best_phot["max_lr"],
        weight_decay=best_phot["weight_decay"],
        epochs=EPOCHS_BEST,
        batch_size=BATCH_SIZE,
        warmup_frac=WARMUP_FRAC,
        run_name="phot_best",
        device=device,
        pos_weight=pos_weight,
        save_best=args.save_best,
        save_path=save_dir / f"best_model_phot_{args.outname}.pt",
    )

    print("Done.")


if __name__ == "__main__":
    main()
