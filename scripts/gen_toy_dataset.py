import argparse, os, numpy as np, h5py
from tqdm import trange

def roll(A, dx, dy): return np.roll(np.roll(A, dx, axis=0), dy, axis=1)

def simulate(n_steps=10, res=128, adv=(0.6,-0.4), diff=0.001, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1,1,res); y = np.linspace(-1,1,res)
    X, Y = np.meshgrid(x,y, indexing='ij')
    S = np.exp(-20*((X-0.2)**2 + (Y+0.1)**2)) + 0.7*np.exp(-15*((X+0.3)**2 + (Y-0.2)**2))
    S += 0.001*rng.standard_normal(S.shape); S = np.clip(S, 0, None)

    u, v = adv
    out = [S.astype(np.float32)]
    for _ in range(n_steps):
        dx, dy = int(np.sign(u)), int(np.sign(v))
        S_adv = 0.9*S + 0.1*roll(S, dx, dy)                    # crude advection
        lap = (-4*S + roll(S,1,0) + roll(S,-1,0) + roll(S,0,1) + roll(S,0,-1))
        S = np.clip(S_adv + diff*lap, 0, None)                 # diffusion
        out.append(S.astype(np.float32))
    return np.stack(out, axis=0)  # [T+1,H,W]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", type=str, default="data/toy_advecdiff.h5")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    Xs, Ys, Ps = [], [], []
    for i in trange(args.n):
        u = np.random.uniform(-0.8,0.8); v = np.random.uniform(-0.8,0.8)
        diff = 10**np.random.uniform(-3.2,-2.2)
        sim = simulate(n_steps=args.steps, res=args.res, adv=(u,v), diff=diff, seed=i)
        Xs.append(sim[:-1])   # [T,H,W]
        Ys.append(sim[1:])    # [T,H,W]
        Ps.append([u,v,diff]) # [3]
    X = np.stack(Xs); Y = np.stack(Ys); P = np.array(Ps, dtype=np.float32)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=X); f.create_dataset("Y", data=Y); f.create_dataset("P", data=P)
    print("Saved", args.out, X.shape, Y.shape, P.shape)

if __name__ == "__main__":
    main()
