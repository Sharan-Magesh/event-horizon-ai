import argparse, os, sys, h5py, torch
import numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# make project root importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.fno2d import FNO2D
from src.physics_losses import total_mass, center_of_brightness


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_grmhd.h5")
    ap.add_argument("--ckpt", default="checkpoints/fno_synth_physics_full.pt")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--modes", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("logs", exist_ok=True)

    # dataset + single test example
    ds = DiskSeq(args.data, "test")
    x_seq, p = ds[0]
    x_seq = x_seq.unsqueeze(0).float().to(device)   # [1,T,H,W]
    p = p.unsqueeze(0).float().to(device)           # [1,3]
    _, T, H, W = x_seq.shape

    horizon = min(args.horizon, T - 1)

    # load model
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
    gt_frames = []
    pred_frames = []

    state = x_seq[:, 0].unsqueeze(1)   # [1,1,H,W]
    gt_frames.append(state.clone())

    with torch.no_grad():
        for t in range(1, horizon + 1):
            gt_t = x_seq[:, t].unsqueeze(1)
            pred_t = model(state, p)
            gt_frames.append(gt_t)
            pred_frames.append(pred_t)
            state = pred_t

    gt_stack = torch.cat(gt_frames, dim=0)        # [h+1,1,H,W]
    pred_stack = torch.cat([gt_frames[0]] + pred_frames, dim=0)

    # ----- compute mass + COM -----
    mass_gt = total_mass(gt_stack).cpu().numpy()
    mass_pred = total_mass(pred_stack).cpu().numpy()

    com_gt = center_of_brightness(gt_stack).cpu().numpy()      # [T,2]
    com_pred = center_of_brightness(pred_stack).cpu().numpy()  # [T,2]

    steps = np.arange(0, horizon + 1)

    # plot mass
    plt.figure(figsize=(5,4), dpi=120)
    plt.plot(steps, mass_gt, label="GT")
    plt.plot(steps, mass_pred, label="Pred", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Total mass (brightness)")
    plt.title("Mass conservation over rollout")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/mass_rollout.png")
    print("Saved logs/mass_rollout.png")

    # plot COM radius + angle
    radius_gt = np.sqrt(com_gt[:,0]**2 + com_gt[:,1]**2)
    radius_pred = np.sqrt(com_pred[:,0]**2 + com_pred[:,1]**2)

    angle_gt = np.arctan2(com_gt[:,1], com_gt[:,0])
    angle_pred = np.arctan2(com_pred[:,1], com_pred[:,0])

    plt.figure(figsize=(10,4), dpi=120)

    plt.subplot(1,2,1)
    plt.plot(steps, radius_gt, label="GT")
    plt.plot(steps, radius_pred, label="Pred", linestyle="--")
    plt.xlabel("Step"); plt.ylabel("Radius")
    plt.title("COM radius over time"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(steps, angle_gt, label="GT")
    plt.plot(steps, angle_pred, label="Pred", linestyle="--")
    plt.xlabel("Step"); plt.ylabel("Angle (rad)")
    plt.title("COM angle over time"); plt.legend()

    plt.tight_layout()
    plt.savefig("logs/com_rollout.png")
    print("Saved logs/com_rollout.png")


if __name__ == "__main__":
    main()
