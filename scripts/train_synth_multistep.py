import argparse, os, sys, h5py, torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# import from src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.fno2d import FNO2D


class DiskSeq(Dataset):
    """
    Sequence dataset for synthetic GRMHD-like disks.
    Returns X[i] of shape [T, H, W] and params P[i].
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
        x_seq = self.X[i]      # [T, H, W]
        p = self.P[i]          # [3]
        return torch.from_numpy(x_seq), torch.from_numpy(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_grmhd.h5")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--modes", type=int, default=48)
    ap.add_argument("--max_horizon", type=int, default=6,
                    help="max rollout steps during training")
    ap.add_argument("--alpha_1step", type=float, default=0.3,
                    help="weight for 1-step loss vs multi-step loss")
    ap.add_argument("--ckpt", default="checkpoints/fno_synth_multistep.pt")
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

    best = 1e9
    horizons_choices = [1, 2, 3, 4, 5, 6]

    for ep in range(args.epochs):
        model.train()
        tr_mse_1step, tr_mse_ms = 0.0, 0.0
        n_tr = 0

        for x_seq, p in tqdm(dl_tr, desc=f"train ep{ep+1}"):
            x_seq = x_seq.float().to(device)   # [B,T,H,W]
            p = p.float().to(device)           # [B,3]
            B, T, H, W = x_seq.shape
            if T < 2:
                continue

            # pick random start time t0
            max_start = max(T - (args.max_horizon + 1), 1)
            t0 = torch.randint(low=0, high=max_start, size=(1,)).item()

            # valid horizons from this t0
            valid_h = [h for h in horizons_choices if t0 + h < T]
            if not valid_h:
                continue
            horizon = int(np.random.choice(valid_h))

            # ---- 1-step loss (local accuracy) from same t0 ----
            inp_1 = x_seq[:, t0].unsqueeze(1)      # [B,1,H,W]
            gt_1 = x_seq[:, t0 + 1].unsqueeze(1)   # [B,1,H,W]
            pred_1 = model(inp_1, p)
            loss_1 = mse_loss(pred_1, gt_1)

            # ---- multi-step loss (rollout-aware) ----
            state = x_seq[:, t0].unsqueeze(1)      # [B,1,H,W]
            loss_ms = 0.0
            
            # scheduled sampling probability: start small, grow over epochs
            ss_prob = float(ep + 1) / float(args.epochs)   # from ~0 → 1
            
            for k in range(1, horizon + 1):
                gt_prev = x_seq[:, t0 + k - 1].unsqueeze(1)   # GT input for this step
                gt_k = x_seq[:, t0 + k].unsqueeze(1)   # [B,1,H,W]
                
                # with probability ss_prob, use model's own previous prediction as input
                # otherwise, use ground truth as input
                use_model = (torch.rand(1, device=device) < ss_prob).item()
                inp_k = state if use_model else gt_prev

                pred_k = model(inp_k, p)
                loss_ms = loss_ms + mse_loss(pred_k, gt_k)
                
                state = pred_k                        # feed prediction forward

            loss_ms = loss_ms / horizon

            loss = args.alpha_1step * loss_1 + (1.0 - args.alpha_1step) * loss_ms

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_mse_1step += loss_1.item() * B
            tr_mse_ms += loss_ms.item() * B
            n_tr += B

        tr_mse_1step /= max(n_tr, 1)
        tr_mse_ms /= max(n_tr, 1)

        # ---- evaluation: 1-step MSE on test set ----
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
            f"train_1step {tr_mse_1step:.6e}  "
            f"train_multistep {tr_mse_ms:.6e}  "
            f"test_1step {te_mse:.6e}"
        )

        if te_mse < best:
            best = te_mse
            torch.save(model.state_dict(), args.ckpt)

    # qualitative 1-step visualization on first test sequence
    x_seq, p = te[0]
    x_seq = x_seq.unsqueeze(0).float().to(device)   # [1,T,H,W]
    p = p.unsqueeze(0).float().to(device)           # [1,3]
    with torch.no_grad():
        inp = x_seq[:, 0].unsqueeze(1)              # [1,1,H,W]
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
