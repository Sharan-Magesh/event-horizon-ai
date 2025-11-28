import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# --------------------------------------------
#  Utility: turbulence via smoothed random noise
# --------------------------------------------
def turbulence(H, W, strength=0.0, smooth=3.0):
    """
    Set strength=0.0 for now to remove noise.
    You can turn it back on later once the FNO learns clean dynamics.
    """
    if strength <= 0.0:
        return np.zeros((H, W), dtype=np.float32)
    noise = np.random.randn(H, W)
    return (strength * gaussian_filter(noise, smooth)).astype(np.float32)


# --------------------------------------------
#  Radial disk profile
# --------------------------------------------
def base_disk(H, W):
    y, x = np.indices((H, W))
    cx, cy = W // 2, H // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / np.max(r)
    disk = np.exp(-4 * r_norm)  # exponential falloff
    return disk.astype(np.float32)


# --------------------------------------------
#  Spiral arms (2–3 spirals)
# --------------------------------------------
def spirals(H, W, t, num_arms=2, winding=5.0, speed=0.1):
    y, x = np.indices((H, W))
    cx, cy = W // 2, H // 2
    angle = np.arctan2(y - cy, x - cx)

    # rotate angle over time
    angle = angle - speed * t

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / np.max(r)

    spiral = np.zeros_like(r_norm)
    for k in range(num_arms):
        target = winding * r_norm + k * (2 * np.pi / num_arms)
        spiral += np.exp(-((angle - target) ** 2) / 0.15)

    # mask spirals mostly to a ring region
    mask = (r_norm > 0.15) & (r_norm < 0.9)
    spiral *= mask
    spiral = spiral / (spiral.max() + 1e-6)
    return spiral.astype(np.float32)


# --------------------------------------------
#  Hotspot orbiting
# --------------------------------------------
def hotspot(H, W, t, radius=0.35, size=4.0, speed=0.25):
    y, x = np.indices((H, W))
    cx, cy = W // 2, H // 2
    theta = speed * t

    hx = cx + radius * W * np.cos(theta)
    hy = cy + radius * H * np.sin(theta)

    spot = np.exp(-(((x - hx) ** 2 + (y - hy) ** 2) / (2 * size ** 2)))
    spot = spot / (spot.max() + 1e-6)
    return spot.astype(np.float32)


# --------------------------------------------
#  Combined synthetic GRMHD-like frame
# --------------------------------------------
def synth_frame(H, W, t, speed=0.12, num_arms=2):
    disk = base_disk(H, W)
    turb = turbulence(H, W, strength=0.0, smooth=4)  # <-- noise off for now
    arms = spirals(H, W, t, num_arms=num_arms, winding=6.0, speed=speed)
    spot = hotspot(H, W, t, radius=0.35, size=4.0, speed=0.6 * speed)

    frame = 0.6 * disk + 0.3 * arms + 0.25 * spot + turb
    frame = np.clip(frame, 0, None)
    frame /= frame.max() + 1e-6
    return frame.astype(np.float32)


# --------------------------------------------
#  Generate dataset
# --------------------------------------------
def generate_dataset(N=50, T=30, H=128, W=128, outpath="data/synth_grmhd.h5"):
    X = np.zeros((N, T, H, W), dtype=np.float32)
    P = np.zeros((N, 3), dtype=np.float32)    # [speed, num_arms, dummy]

    for i in range(N):
        # randomize parameters per sequence
        speed = np.random.uniform(0.08, 0.16)
        arms = int(np.random.choice([2, 3]))
        P[i] = [speed, float(arms), 1.0]

        for t in range(T):
            X[i, t] = synth_frame(H, W, t * speed, speed=speed, num_arms=arms)

    with h5py.File(outpath, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("P", data=P)

    print("Saved:", outpath)


# --------------------------------------------
#  Preview
# --------------------------------------------
if __name__ == "__main__":
    H = W = 128
    frame = synth_frame(H, W, t=0.0)
    plt.figure(figsize=(5, 5), dpi=120)
    plt.imshow(frame, cmap="inferno")
    plt.colorbar()
    plt.title("Synthetic GRMHD-Like Frame (t=0)")
    plt.tight_layout()
    plt.show()

    generate_dataset()
