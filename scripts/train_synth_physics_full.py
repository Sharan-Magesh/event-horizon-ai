import argparse, os, sys, h5py, torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.fno2d import FNO2D
from src.physics_losses import physics_loss_full


# -----------------------------
# Dataset
# -----------------------------
class DiskSeq(Dataset):
    """
    Sequence dataset for synthetic GRMHD-like disks.
    Returns X[i] of shape [T,H,W] and params P[i].
    """
    def __init__(self, path, split="train", ratio=0.9):
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


# -----------------------------
# Training
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_grmhd.h5")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--modes", type=int, default=48)
    ap.add_argument("--max_horizon", type=int, default=8)
    ap.add_argument("--alpha_1step", type=float, default=0.25,
                    help="weight for 1-step loss vs multi-step")

    # physics prior weights (slightly stronger than before)
    ap.add_argument("--lambda_mass", type=float, default=1e-2)
    ap.add_argument("--lambda_radial", type=float, default=5e-3)
    ap.add_argument("--lambda_com", type=float, default=5e-3)
    ap.add_argument("--lambda_grad", type=float, default=1e-3)
    ap.add_argument("--lambda_lap", type=float, default=1e-3)

    ap.add_argument("--ckpt", default="checkpoints/fno_synth_physics_full.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    tr = DiskSeq(args.data, "train")
    te = DiskSeq(args.data, "test")
    dl_tr = DataLoader(tr, batch_size=args.bs, shuffle=True)
    dl_te = DataLoader(te, batch_size=1, shuffle=False)

    model = FNO2D(
        in_channels=1,
        width=args.width,
        depth=4,
        modes1=args.modes,
        modes2=args.modes,
        param_channels=3,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    # precompute radius map (normalized 0..1)
    with h5py.File(args.data, "r") as f:
        H, W = f["X"].shape[-2:]
    y, x = np.indices((H, W))
    cx, cy = W // 2, H // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r / r.max()
    r_map = torch.from_numpy(r.astype(np.float32))   # [H,W]

    best = 1e9
    horizons_choices = [1, 2, 3, 4, 5, 6, 7, 8]

    for ep in range(args.epochs):
        model.train()
        tr_mse_1, tr_mse_ms, tr_phys = 0.0, 0.0, 0.0
        n_tr = 0

        for x_seq, p in tqdm(dl_tr, desc=f"train ep{ep+1}"):
            x_seq = x_seq.float().to(device)   # [B,T,H,W]
            p = p.float().to(device)           # [B,3]
            B, T, H, W = x_seq.shape
            if T < 2:
                continue

            # random start time
            max_start = max(T - (args.max_horizon + 1), 1)
            t0 = torch.randint(low=0, high=max_start, size=(1,)).item()

            # valid horizons from this t0
            valid_h = [h for h in horizons_choices if t0 + h < T]
            if not valid_h:
                continue
            horizon = int(np.random.choice(valid_h))

            # ----- 1-step loss -----
            inp_1 = x_seq[:, t0].unsqueeze(1)       # [B,1,H,W]
            gt_1 = x_seq[:, t0 + 1].unsqueeze(1)
            pred_1 = model(inp_1, p)
            loss_1 = mse_loss(pred_1, gt_1)

            # ----- multi-step + physics -----
            state = x_seq[:, t0].unsqueeze(1)
            loss_ms = 0.0
            loss_phys = 0.0

            # scheduled sampling prob: 0 -> 1 over epochs
            ss_prob = float(ep + 1) / float(args.epochs)

            for k in range(1, horizon + 1):
                gt_prev = x_seq[:, t0 + k - 1].unsqueeze(1)  # [B,1,H,W]
                gt_k = x_seq[:, t0 + k].unsqueeze(1)

                # choose input: model state vs ground-truth previous
                use_model = (torch.rand(1, device=device) < ss_prob).item()
                inp_k = state if use_model else gt_prev

                pred_k = model(inp_k, p)

                # core multi-step MSE
                l_mse = mse_loss(pred_k, gt_k)
                loss_ms = loss_ms + l_mse

                # full physics prior (mass + radial + COM + grad + Laplacian)
                l_phys = physics_loss_full(
                    pred_k,
                    gt_k,
                    r_map,
                    lambda_mass=args.lambda_mass,
                    lambda_radial=args.lambda_radial,
                    lambda_com=args.lambda_com,
                    lambda_grad=args.lambda_grad,
                    lambda_lap=args.lambda_lap,
                    nbins=32,
                )
                loss_phys = loss_phys + l_phys

                state = pred_k

            loss_ms = loss_ms / horizon
            loss_phys = loss_phys / horizon

            loss = (
                args.alpha_1step * loss_1
                + (1.0 - args.alpha_1step) * loss_ms
                + loss_phys
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_mse_1 += loss_1.item() * B
            tr_mse_ms += loss_ms.item() * B
            tr_phys += loss_phys.item() * B
            n_tr += B

        tr_mse_1 /= max(n_tr, 1)
        tr_mse_ms /= max(n_tr, 1)
        tr_phys /= max(n_tr, 1)

        # ---- evaluation: plain 1-step MSE ----
        model.eval()
        te_mse = 0.0
        n_te = 0
        with torch.no_grad():
            for x_seq, p in dl_te:
                x_seq = x_seq.float().to(device)
                p = p.float().to(device)
                B, T, H, W = x_seq.shape
                if T < 2:
                    continue
                inp = x_seq[:, 0].unsqueeze(1)
                gt = x_seq[:, 1].unsqueeze(1)
                pred = model(inp, p)
                mse = mse_loss(pred, gt)
                te_mse += mse.item() * B
                n_te += B
        te_mse /= max(n_te, 1)

        print(
            f"Epoch {ep+1}: "
            f"train_1step {tr_mse_1:.6e}  "
            f"train_multistep {tr_mse_ms:.6e}  "
            f"train_phys {tr_phys:.6e}  "
            f"test_1step {te_mse:.6e}"
        )

        if te_mse < best:
            best = te_mse
            torch.save(model.state_dict(), args.ckpt)

    # ---- qualitative 1-step visualization ----
    x_seq, p = te[0]
    x_seq = x_seq.unsqueeze(0).float().to(device)
    p = p.unsqueeze(0).float().to(device)
    with torch.no_grad():
        inp = x_seq[:, 0].unsqueeze(1)
        gt = x_seq[:, 1].unsqueeze(1)
        pred = model(inp, p).cpu()[0, 0].numpy()

    inp_np = x_seq[0, 0].cpu().numpy()
    gt_np = x_seq[0, 1].cpu().numpy()

    plt.figure(figsize=(9, 3), dpi=120)
    for i, (im, title) in enumerate(
        [(inp_np, "Input S_t"), (gt_np, "GT S_{t+Δt}"), (pred, "Pred S_{t+Δt}")], 1
    ):
        plt.subplot(1, 3, i)
        plt.imshow(im)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("logs/qual_toy.png")
    print("Saved logs/qual_toy.png")


if __name__ == "__main__":
    main()
