import torch
import torch.nn as nn


# =============================
#   GEOMETRIC UTILITIES
# =============================

def total_mass(field):
    """
    field: [B,1,H,W]
    returns: [B] total intensity per sample
    """
    return field.sum(dim=(1, 2, 3))


def radial_profile(field, r_map, nbins=32):
    """
    field: [B,1,H,W]
    r_map: [H,W] with radius (0..1) from center (precomputed)
    returns: [B, nbins] radial average profile
    """
    B, _, H, W = field.shape
    device = field.device
    r = r_map.to(device)  # [H,W]

    # bin indices 0..nbins-1
    idx = torch.clamp((r * nbins).long(), 0, nbins - 1)  # [H,W]

    # flatten
    f_flat = field.view(B, -1)         # [B,HW]
    idx_flat = idx.view(-1)            # [HW]

    prof = torch.zeros(B, nbins, device=device)
    counts = torch.zeros(nbins, device=device)

    for b in range(B):
        prof[b].index_add_(0, idx_flat, f_flat[b])
    counts.index_add_(0, idx_flat,
                      torch.ones_like(idx_flat, dtype=torch.float32, device=device))

    counts = torch.clamp(counts, min=1.0)
    prof = prof / counts.unsqueeze(0)  # [B,nbins]
    return prof


def center_of_brightness(field):
    """
    field: [B,1,H,W]
    returns: [B,2] (cx, cy) in normalized coords [-1,1]
    """
    B, _, H, W = field.shape
    device = field.device

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    x = (x.float() / (W - 1)) * 2 - 1  # [-1,1]
    y = (y.float() / (H - 1)) * 2 - 1  # [-1,1]

    f = field.squeeze(1)               # [B,H,W]
    mass = torch.clamp(f.sum(dim=(1, 2), keepdim=True), min=1e-6)  # [B,1]

    cx = (f * x).sum(dim=(1, 2), keepdim=True) / mass
    cy = (f * y).sum(dim=(1, 2), keepdim=True) / mass

    return torch.cat([cx, cy], dim=1)  # [B,2]


# =============================
#   GRADIENT / LAPLACIAN
# =============================

def gradient(field):
    """
    Simple finite-difference gradient.
    field: [B,1,H,W]
    returns: (gx, gy) each [B,1,H,W]
    """
    # pad by replication at borders
    field_pad_x = torch.nn.functional.pad(field, (1, 1, 0, 0), mode="replicate")
    field_pad_y = torch.nn.functional.pad(field, (0, 0, 1, 1), mode="replicate")

    gx = field_pad_x[:, :, :, 2:] - field_pad_x[:, :, :, :-2]  # diff in x
    gy = field_pad_y[:, :, 2:, :] - field_pad_y[:, :, :-2, :]  # diff in y
    return gx * 0.5, gy * 0.5


def grad_magnitude(field):
    gx, gy = gradient(field)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def laplacian(field):
    """
    5-point stencil Laplacian.
    field: [B,1,H,W]
    returns: [B,1,H,W]
    """
    B, C, H, W = field.shape
    # pad by replication
    f = torch.nn.functional.pad(field, (1, 1, 1, 1), mode="replicate")
    center = f[:, :, 1:-1, 1:-1]
    up = f[:, :, :-2, 1:-1]
    down = f[:, :, 2:, 1:-1]
    left = f[:, :, 1:-1, :-2]
    right = f[:, :, 1:-1, 2:]
    return (up + down + left + right - 4.0 * center)


# =============================
#   BASIC PHYSICS LOSSES
# =============================

def mass_loss(pred, gt):
    """
    Relative mass / brightness conservation loss.
    pred, gt: [B,1,H,W]
    """
    m_pred = total_mass(pred)
    m_gt = total_mass(gt)
    rel_err = (m_pred - m_gt) / (m_gt + 1e-6)
    return (rel_err ** 2).mean()


def radial_profile_loss(pred, gt, r_map, nbins=32):
    """
    Match radial brightness structure.
    """
    mse = nn.MSELoss()
    rp_pred = radial_profile(pred, r_map, nbins=nbins)
    rp_gt = radial_profile(gt, r_map, nbins=nbins)
    return mse(rp_pred, rp_gt)


def com_loss(pred, gt):
    """
    Center-of-brightness (hotspot / crescent location) consistency.
    """
    mse = nn.MSELoss()
    com_pred = center_of_brightness(pred)
    com_gt = center_of_brightness(gt)
    return mse(com_pred, com_gt)


def grad_loss(pred, gt):
    """
    Match gradient magnitude (edge strength / sharpness).
    """
    mse = nn.MSELoss()
    g_pred = grad_magnitude(pred)
    g_gt = grad_magnitude(gt)
    return mse(g_pred, g_gt)


def laplacian_loss(pred, gt):
    """
    Match curvature structure (disc / ring thickness).
    """
    mse = nn.MSELoss()
    lap_pred = laplacian(pred)
    lap_gt = laplacian(gt)
    return mse(lap_pred, lap_gt)


# =============================
#   COMBINED HELPERS
# =============================

def physics_loss_basic(
    pred,
    gt,
    r_map,
    lambda_mass=1e-2,
    lambda_radial=5e-3,
    lambda_com=5e-3,
    nbins=32,
):
    """
    Lightweight physics prior:
      - mass conservation
      - radial profile consistency
      - center-of-brightness consistency
    """
    l = 0.0
    if lambda_mass > 0:
        l += lambda_mass * mass_loss(pred, gt)
    if lambda_radial > 0:
        l += lambda_radial * radial_profile_loss(pred, gt, r_map, nbins=nbins)
    if lambda_com > 0:
        l += lambda_com * com_loss(pred, gt)
    return l


def physics_loss_full(
    pred,
    gt,
    r_map,
    lambda_mass=1e-2,
    lambda_radial=5e-3,
    lambda_com=5e-3,
    lambda_grad=1e-3,
    lambda_lap=1e-3,
    nbins=32,
):
    """
    Full physics-ish prior for this synthetic GRMHD toy:
      - mass conservation
      - radial brightness profile consistency
      - center-of-brightness (rotation) consistency
      - gradient magnitude consistency (edge structure of the crescent)
      - Laplacian consistency (curvature / thickness of ring)
    """
    l = physics_loss_basic(
        pred,
        gt,
        r_map,
        lambda_mass=lambda_mass,
        lambda_radial=lambda_radial,
        lambda_com=lambda_com,
        nbins=nbins,
    )

    if lambda_grad > 0:
        l += lambda_grad * grad_loss(pred, gt)

    if lambda_lap > 0:
        l += lambda_lap * laplacian_loss(pred, gt)

    return l
