import argparse, os, sys, h5py, torch
import numpy as np
import matplotlib.pyplot as plt

# import from src
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.fno2d import FNO2D


def load_sample(path, test_ratio=0.1, index=0):
    with h5py.File(path, "r") as f:
        X = f["X"][:]  # [N,T,H,W]
        P = f["P"][:]
    N = X.shape[0]
    n_test = int(N * test_ratio)
    start = N - n_test
    i = start + index
    seq_X = X[i]       # [T,H,W]
    params = P[i]      # [3]

    # build GT sequence S_0..S_T (here just use X directly)
    S_gt = seq_X       # [T,H,W]; we will interpret step indices accordingly
    return S_gt, params


def rollout_model(model, S0, params, horizon, device):
    """
    S0: [H,W] numpy
    params: [3] numpy
    returns: [horizon+1, H, W] numpy (includes initial state)
    """
    model.eval()
    H, W = S0.shape
    cur = torch.from_numpy(S0[None, None]).float().to(device)  # [1,1,H,W]
    p = torch.from_numpy(params[None]).float().to(device)      # [1,3]
    preds = [S0.copy()]
    with torch.no_grad():
        for _ in range(horizon):
            nxt = model(cur, p)
            cur = nxt
            preds.append(nxt.cpu().numpy()[0, 0])
    return np.stack(preds, axis=0)  # [horizon+1,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/toy_advecdiff.h5")
    ap.add_argument("--ckpt", default="checkpoints/fno_toy.pt")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("logs", exist_ok=True)

    S_gt, params = load_sample(args.data, index=args.sample)   # [T,H,W]
    T_gt, H, W = S_gt.shape
    horizon = min(args.horizon, T_gt - 1)

    model = FNO2D(
        in_channels=1,
        width=args.width,
        depth=4,
        modes1=args.modes,
        modes2=args.modes,
        param_channels=3,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    S_pred = rollout_model(model, S_gt[0], params, horizon, device)

    # error per step (1..horizon)
    step_errors = []
    for t in range(1, horizon + 1):
        mse = np.mean((S_pred[t] - S_gt[t]) ** 2)
        step_errors.append(mse)

    # visualization: GT vs Pred at a few steps
    times = np.linspace(0, horizon, num=min(5, horizon + 1), dtype=int)
    plt.figure(figsize=(len(times) * 3, 6), dpi=120)
    for i, t in enumerate(times):
        plt.subplot(2, len(times), i + 1)
        plt.imshow(S_gt[t])
        plt.title(f"GT t={t}")
        plt.axis("off")

        plt.subplot(2, len(times), len(times) + i + 1)
        plt.imshow(S_pred[t])
        plt.title(f"Pred t={t}")
        plt.axis("off")

    plt.tight_layout()
    rollout_path = "logs/rollout_toy.png"
    plt.savefig(rollout_path)
    print("Saved", rollout_path)

    # error plot
    steps = np.arange(1, horizon + 1)
    plt.figure(figsize=(5, 4), dpi=120)
    plt.plot(steps, step_errors, marker="o")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.title("Rollout prediction error vs time")
    plt.grid(True, alpha=0.3)
    err_path = "logs/rollout_error_toy.png"
    plt.tight_layout()
    plt.savefig(err_path)
    print("Saved", err_path)


if __name__ == "__main__":
    main()
