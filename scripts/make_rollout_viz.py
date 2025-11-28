import argparse
import os
import sys

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.fno2d import FNO2D  # your existing model


# ---------------------------------------------------------------------
# Dataset (same idea as eval_physics_metrics.py)
# ---------------------------------------------------------------------
class DiskSeq(Dataset):
    def __init__(self, path, split="test", ratio=0.9):
        with h5py.File(path, "r") as f:
            X, P = f["X"][:], f["P"][:]
        N = X.shape[0]
        ntr = int(N * ratio)
        if split == "train":
            self.X, self.P = X[:ntr], P[:ntr]
        else:
            self.X, self.P = X[ntr:], P[ntr:]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x_seq = self.X[i]      # [T,H,W]
        p = self.P[i]          # [3]
        return torch.from_numpy(x_seq), torch.from_numpy(p)


# ---------------------------------------------------------------------
# Rollout helper
# ---------------------------------------------------------------------
def rollout_sequence(model, x_seq, p, horizon):
    """
    x_seq: [1,T,H,W]
    p:     [1,3]
    horizon: int (<= T-1)
    Returns:
        gt_stack:   [T', H, W]
        pred_stack: [T', H, W]
    where T' = horizon+1
    """
    device = x_seq.device
    _, T, H, W = x_seq.shape
    horizon = min(horizon, T - 1)

    # ground-truth frames 0..horizon
    gt_stack = x_seq[0, :horizon + 1].cpu().numpy()  # [T',H,W]

    # autoregressive prediction
    state = x_seq[:, 0].unsqueeze(1)  # [1,1,H,W]
    preds = [state[0, 0].detach().cpu().numpy()]
    with torch.no_grad():
        for t in range(1, horizon + 1):
            state = model(state, p)  # [1,1,H,W]
            preds.append(state[0, 0].detach().cpu().numpy())
    pred_stack = np.stack(preds, axis=0)  # [T',H,W]

    return gt_stack, pred_stack


# ---------------------------------------------------------------------
# Frame rendering helpers
# ---------------------------------------------------------------------
def render_frame_grid(gt, pred, vmin=None, vmax=None):
    """
    2x1 grid: top GT, bottom Pred
    Returns RGB image array.
    """
    fig, axes = plt.subplots(2, 1, figsize=(4, 6))
    for ax in axes:
        ax.axis("off")

    im0 = axes[0].imshow(gt, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("GT")

    im1 = axes[1].imshow(pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Pred")

    fig.tight_layout(pad=0.1)

    # draw and grab as array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)
    buf = buf[:, :, 1:]  # drop alpha, keep RGB
    img = buf.copy()
    plt.close(fig)
    return img


def render_frame_side_by_side(gt, pred, vmin=None, vmax=None):
    """
    1x2 grid: left GT, right Pred
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(gt, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("GT")

    axes[1].imshow(pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Pred")

    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)
    buf = buf[:, :, 1:]
    img = buf.copy()
    plt.close(fig)
    return img


def render_frame_triplet(gt, pred, err, vmin=None, vmax=None, err_vmin=None, err_vmax=None):
    """
    1x3 grid: GT | Pred | |GT-Pred|
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(gt, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("GT")

    axes[1].imshow(pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Pred")

    im2 = axes[2].imshow(err, origin="lower", cmap="magma",
                         vmin=err_vmin, vmax=err_vmax)
    axes[2].set_title("|GT - Pred|")

    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)
    buf = buf[:, :, 1:]
    img = buf.copy()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_grmhd.h5")
    ap.add_argument("--ckpt", default="checkpoints/fno_synth_physics_full.pt")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--modes", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--index", type=int, default=0, help="which test sample to visualize")
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)

    # dataset
    ds = DiskSeq(args.data, "test")
    x_seq, p = ds[args.index]
    x_seq = x_seq.unsqueeze(0).float().to(device)  # [1,T,H,W]
    p = p.unsqueeze(0).float().to(device)

    # model
    model = FNO2D(
        in_channels=1,
        width=args.width,
        depth=4,
        modes1=args.modes,
        modes2=args.modes,
        param_channels=3,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # rollout
    gt_stack, pred_stack = rollout_sequence(model, x_seq, p, args.horizon)
    T, H, W = gt_stack.shape

    # global vmin/vmax for consistent color scales
    vmin = float(min(gt_stack.min(), pred_stack.min()))
    vmax = float(max(gt_stack.max(), pred_stack.max()))
    err_stack = np.abs(gt_stack - pred_stack)
    err_vmin, err_vmax = float(err_stack.min()), float(err_stack.max())

    frames_tb = []   # top-bottom GT/Pred
    frames_lr = []   # left-right GT/Pred
    frames_trip = [] # GT | Pred | Error
    frames_pred = [] # Pred only for MP4

    for t in range(T):
        gt = gt_stack[t]
        pred = pred_stack[t]
        err = np.abs(gt - pred)

        frames_tb.append(render_frame_grid(gt, pred, vmin, vmax))
        frames_lr.append(render_frame_side_by_side(gt, pred, vmin, vmax))
        frames_trip.append(
            render_frame_triplet(gt, pred, err, vmin, vmax, err_vmin, err_vmax)
        )

        # pred-only frame for MP4 (simple imshow)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")
        ax.imshow(pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Predicted t={t}")
        fig.tight_layout(pad=0.1)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        buf = buf.reshape(h, w, 4)
        buf = buf[:, :, 1:]
        img = buf.copy()
        plt.close(fig)
        frames_pred.append(img)

    # ----------------- write GIFs -----------------
    gif_tb_path = "results/rollout_gt_pred_topbottom.gif"
    gif_lr_path = "results/rollout_gt_pred_sidebyside.gif"
    gif_trip_path = "results/rollout_gt_pred_error.gif"
    gif_pred_path = "results/rollout_pred_only.gif"

    imageio.mimsave(gif_tb_path, frames_tb, fps=args.fps)
    print(f"Saved {gif_tb_path}")

    imageio.mimsave(gif_lr_path, frames_lr, fps=args.fps)
    print(f"Saved {gif_lr_path}")

    imageio.mimsave(gif_trip_path, frames_trip, fps=args.fps)
    print(f"Saved {gif_trip_path}")

    imageio.mimsave(gif_pred_path, frames_pred, fps=args.fps)
    print(f"Saved {gif_pred_path}")

    # ----------------- write MP4 (pred only) -----------------
    mp4_path = "results/rollout_pred_only.mp4"
    try:
        imageio.mimsave(mp4_path, frames_pred, fps=args.fps, codec="libx264")
        print(f"Saved {mp4_path}")
    except Exception as e:
        print(f"Could not write MP4 (maybe no ffmpeg?): {e}")
        print("GIFs are still saved and usable.")


if __name__ == "__main__":
    main()
