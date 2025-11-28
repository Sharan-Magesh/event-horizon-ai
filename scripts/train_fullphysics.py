import argparse
import os
import sys

import h5py
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.fno2d import FNO2D
from src.physics_losses import total_mass, center_of_brightness


# ---------------------------------------------------------------------
# Dataset
#   X: [N, T, H, W]
#   P: [N, 3]  (e.g. spin, accretion rate, inclination)
# ---------------------------------------------------------------------
class DiskSequence(Dataset):
    def __init__(self, path, split="train", ratio=0.9):
        super().__init__()
        with h5py.File(path, "r") as f:
            X = f["X"][:]  # [N, T, H, W]
            P = f["P"][:]  # [N, 3]
        N = X.shape[0]
        ntr = int(N * ratio)

        if split == "train":
            self.X = X[:ntr]
            self.P = P[:ntr]
        else:
            self.X = X[ntr:]
            self.P = P[ntr:]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_seq = self.X[idx]      # [T, H, W]
        p = self.P[idx]          # [3]
        # convert to tensors
        x_seq = torch.from_numpy(x_seq).float()    # [T,H,W]
        p = torch.from_numpy(p).float()           # [3]
        return x_seq, p


# ---------------------------------------------------------------------
# Multistep rollout + physics losses
# ---------------------------------------------------------------------
def rollout_multistep(model, x_seq, p, horizon):
    """
    x_seq: [B, T, H, W]
    p:     [B, 3]
    horizon: int, number of future steps to predict (<= T-1)

    Returns:
        gt_stack:    [B, horizon+1, 1, H, W]
        pred_stack:  [B, horizon+1, 1, H, W]
        mse_1step:   scalar
        mse_multistep: scalar
    """
    device = x_seq.device
    B, T, H, W = x_seq.shape
    horizon = min(horizon, T - 1)

    # ground-truth frames we care about: 0..horizon
    gt_stack = x_seq[:, :horizon + 1]  # [B, h+1, H, W]
    gt_stack = gt_stack.unsqueeze(2)   # [B, h+1, 1, H, W]

    # 1-step prediction from frame 0 to 1
    x0 = x_seq[:, 0].unsqueeze(1)      # [B,1,H,W]
    gt_next = x_seq[:, 1].unsqueeze(1) # [B,1,H,W]
    pred_1 = model(x0, p)              # [B,1,H,W]
    mse_1step = nn.functional.mse_loss(pred_1, gt_next)

    # autoregressive multistep rollout
    preds = [x0]  # include t=0 frame for alignment
    state = x0
    with torch.no_grad():
        pass
    # we want gradients through rollout, so no torch.no_grad() here
    for t in range(1, horizon + 1):
        state = model(state, p)        # [B,1,H,W]
        preds.append(state)

    pred_stack = torch.stack(preds, dim=1)  # [B, h+1, 1, H, W]

    # multistep mse over all future steps (1..horizon)
    mse_multistep = nn.functional.mse_loss(
        pred_stack[:, 1:],  # [B,h,1,H,W]
        gt_stack[:, 1:]
    )

    return gt_stack, pred_stack, mse_1step, mse_multistep


def physics_losses(gt_stack, pred_stack):
    """
    gt_stack, pred_stack: [B, T, 1, H, W]
    Returns:
        mass_loss, radial_loss, com_loss
    """
    # flatten batch + time to compute global metrics
    B, T, _, H, W = gt_stack.shape
    gt_flat = gt_stack.reshape(B * T, 1, H, W)
    pred_flat = pred_stack.reshape(B * T, 1, H, W)

    # total mass (brightness)
    mass_gt = total_mass(gt_flat)     # [B*T]
    mass_pred = total_mass(pred_flat) # [B*T]
    mass_loss = nn.functional.mse_loss(mass_pred, mass_gt)

    # center of brightness -> radius, (x,y)
    com_gt = center_of_brightness(gt_flat)      # [B*T, 2]
    com_pred = center_of_brightness(pred_flat)  # [B*T, 2]

    radius_gt = torch.sqrt(com_gt[:, 0] ** 2 + com_gt[:, 1] ** 2)
    radius_pred = torch.sqrt(com_pred[:, 0] ** 2 + com_pred[:, 1] ** 2)
    radial_loss = nn.functional.mse_loss(radius_pred, radius_gt)

    com_loss = nn.functional.mse_loss(com_pred, com_gt)

    return mass_loss, radial_loss, com_loss


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # data
    train_ds = DiskSequence(args.data, split="train", ratio=args.train_ratio)
    val_ds = DiskSequence(args.data, split="val", ratio=args.train_ratio)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # model
    model = FNO2D(
        in_channels=1,
        width=args.width,
        depth=args.depth,
        modes1=args.modes,
        modes2=args.modes,
        param_channels=3,
    ).to(device)

    if args.ckpt_init and os.path.isfile(args.ckpt_init):
        print(f"Loading init weights from {args.ckpt_init}")
        state = torch.load(args.ckpt_init, map_location=device)
        model.load_state_dict(state)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, args.epochs // 3),
        gamma=0.5,
    )

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x_seq, p in pbar:
            x_seq = x_seq.to(device)           # [B,T,H,W]
            p = p.to(device)                   # [B,3]

            optimizer.zero_grad()

            gt_stack, pred_stack, mse_1, mse_multi = rollout_multistep(
                model, x_seq, p, args.horizon
            )

            mass_loss, radial_loss, com_loss = physics_losses(gt_stack, pred_stack)

            loss = (
                args.alpha_1step * mse_1
                + args.alpha_multistep * mse_multi
                + args.lambda_mass * mass_loss
                + args.lambda_radial * radial_loss
                + args.lambda_com * com_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x_seq.size(0)

            pbar.set_postfix(
                loss=loss.item(),
                mse1=mse_1.item(),
                msem=mse_multi.item(),
            )

        train_loss /= len(train_ds)

        # ---------------- validation ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq, p in val_loader:
                x_seq = x_seq.to(device)
                p = p.to(device)

                gt_stack, pred_stack, mse_1, mse_multi = rollout_multistep(
                    model, x_seq, p, args.horizon
                )
                mass_loss, radial_loss, com_loss = physics_losses(
                    gt_stack, pred_stack
                )
                loss = (
                    args.alpha_1step * mse_1
                    + args.alpha_multistep * mse_multi
                    + args.lambda_mass * mass_loss
                    + args.lambda_radial * radial_loss
                    + args.lambda_com * com_loss
                )
                val_loss += loss.item() * x_seq.size(0)

        val_loss /= len(val_ds)
        scheduler.step()

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.ckpt_out)
            print(f"  -> Saved best model to {args.ckpt_out} (val_loss={best_val:.6f})")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, default="data/synth_grmhd.h5")
    ap.add_argument("--ckpt_init", type=str, default="", help="optional C1 checkpoint")
    ap.add_argument(
        "--ckpt_out",
        type=str,
        default="checkpoints/fno_synth_fullphysics.pt",
    )

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_ratio", type=float, default=0.9)

    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--modes", type=int, default=48)
    ap.add_argument("--depth", type=int, default=4)

    ap.add_argument("--horizon", type=int, default=6)

    # physics weights (C2)
    ap.add_argument("--alpha_1step", type=float, default=0.25)
    ap.add_argument("--alpha_multistep", type=float, default=0.15)
    ap.add_argument("--lambda_mass", type=float, default=2e-2)
    ap.add_argument("--lambda_radial", type=float, default=1e-2)
    ap.add_argument("--lambda_com", type=float, default=5e-3)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
