"""
Multi-view normal consistency loss for TSGS (MNE module).

Ported from MND-GS (Multi-view Normal Enhancement).
Uses L1 distance between depth-derived normals in world coordinates,
with image-gradient edge-aware weighting and outlier rejection.

Core formula (from MND-GS):
    L_mne = mean[ edge_w^2 * |n_ref^world - n_tgt^world|_1 ]
where edge_w = (1 - image_gradient) * (l1_diff < 0.52)
"""

import torch
import torch.nn.functional as F


# -- helpers -----------------------------------------------------------------

def get_img_grad_weight(img, beta=2.0):
    """Compute image gradient magnitude for edge-aware weighting (from MND-GS).

    Args:
        img: (3, H, W) RGB image tensor.

    Returns:
        (H, W) gradient magnitude normalized to [0, 1].
        Edges -> 1, flat areas -> 0.  Padded to match input size.
    """
    _, hd, wd = img.shape
    bottom = img[..., 2:hd,   1:wd-1]
    top    = img[..., 0:hd-2, 1:wd-1]
    right  = img[..., 1:hd-1, 2:wd]
    left   = img[..., 1:hd-1, 0:wd-2]
    grad_x = torch.mean(torch.abs(right - left), 0, keepdim=True)
    grad_y = torch.mean(torch.abs(top - bottom), 0, keepdim=True)
    grad_img = torch.cat((grad_x, grad_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    eps = 1e-10
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min() + eps)
    grad_img = F.pad(grad_img[None, None], (1, 1, 1, 1), mode='constant', value=1.0).squeeze()
    return grad_img


def _pixels_warp(H, uv):
    """Warp pixel coordinates by a homography matrix (batch or single)."""
    N = uv.shape[0]
    ones = torch.ones((N, 1), device=uv.device, dtype=uv.dtype)
    homo_uv = torch.cat((uv, ones), dim=-1)                     # (N, 3)
    grid_tmp = (H @ homo_uv.unsqueeze(-1)).squeeze(-1)           # (N, 3)
    grid = grid_tmp[:, :2] / (grid_tmp[:, 2:] + 1e-10)
    return grid


def _sample_normal_map(normal_map, pixel_coords, H, W):
    """Bilinear-sample a (3, H, W) normal map at given pixel positions."""
    grid = pixel_coords.view(1, -1, 1, 2)
    sampled = F.grid_sample(
        normal_map.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    return sampled[0, :, :, 0]  # (3, N)


# -- main API ---------------------------------------------------------------

def compute_mv_normal_loss(
    viewpoint_cam,
    nearest_cam,
    render_pkg,
    nearest_render_pkg,
    valid_indices,
    H_ref_to_neareast,
    ix, iy,
    H, W,
    mv_normal_weight=0.03,
    loss_type="l1",
    normal_key="depth_normal",
    max_samples=0,
    reliability_weights=None,
    edge_weight=None,
    reliability_for_sampling=None,
):
    """Compute multi-view normal consistency loss (MNE from MND-GS).

    Args:
        viewpoint_cam:       Reference camera.
        nearest_cam:         Target (neighbour) camera.
        render_pkg:          Dict from render(viewpoint_cam, ...).
        nearest_render_pkg:  Dict from render(nearest_cam, ...).
        valid_indices:       (M,) long tensor -- geometry-consistent pixel indices.
        H_ref_to_neareast:   (M, 3, 3) per-pixel homography from ref -> target.
        ix, iy:              Meshgrid outputs (W, H) in xy indexing.
        H, W:                Image height and width.
        mv_normal_weight:    Scalar weight (applied outside by caller).
        loss_type:           "l1" (default, MNE style) or "cosine".
        normal_key:          Which normal from render_pkg ("depth_normal" or
                             "rendered_normal").
        max_samples:         If > 0, randomly subsample to this many pixels.
                             0 = use all valid_indices.
        reliability_weights: (M,) optional R-map weights (legacy, prefer edge_weight).
        edge_weight:         (M,) from (1 - image_gradient), flat areas -> 1,
                             edges -> 0.  Used for MNE-style weighting.
        reliability_for_sampling: (M,) R-map values in [0,1] for inverse
                             reliability sampling.  Low-R (transparent) pixels
                             are oversampled so MNE focuses on regions that
                             need the most geometric correction.

    Returns:
        dict with keys: loss, has_valid, n_valid, cos_sim, ok_mask,
                        loss_per_pixel, weights, pixel_indices.
    """
    result = {"loss": torch.tensor(0.0, device="cuda"), "has_valid": False,
              "n_valid": 0, "cos_sim": None, "ok_mask": None,
              "loss_per_pixel": None, "weights": None, "pixel_indices": None}

    if valid_indices.numel() == 0:
        return result

    # ---- optional subsampling ----
    _sampling_diag = {}  # [DIAG] MNE sampling diagnostics
    _needs_subsample = max_samples > 0 and valid_indices.numel() > max_samples
    if _needs_subsample:
        if reliability_for_sampling is not None:
            # Inverse reliability sampling: oversample transparent (low-R)
            # pixels so MNE concentrates correction where it matters most.
            # No circular dependency: R guides *where* to sample, not
            # *how much* gradient flows -- negative-feedback convergent.
            _pop_R = reliability_for_sampling.detach()  # [DIAG]
            inv_r = (1.0 - _pop_R) + 0.5
            prob = inv_r / inv_r.sum()
            sel = torch.multinomial(prob, max_samples, replacement=False)
            # [DIAG] Record population vs sampled R distribution
            _sampled_R = _pop_R[sel]
            _sampling_diag = {
                "population_R_mean": _pop_R.mean().item(),
                "sampled_R_mean": _sampled_R.mean().item(),
                "sampling_bias": _pop_R.mean().item() - _sampled_R.mean().item(),
                "population_R_lt03": (_pop_R < 0.3).float().mean().item(),
                "sampled_R_lt03": (_sampled_R < 0.3).float().mean().item(),
                "population_n": int(_pop_R.numel()),
                "sampled_n": int(_sampled_R.numel()),
            }
        else:
            sel = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:max_samples]
        valid_indices = valid_indices[sel]
        H_ref_to_neareast = H_ref_to_neareast[sel]
        if reliability_weights is not None:
            reliability_weights = reliability_weights[sel]
        if edge_weight is not None:
            edge_weight = edge_weight[sel]
        if reliability_for_sampling is not None:
            reliability_for_sampling = reliability_for_sampling[sel]
    elif reliability_for_sampling is not None:
        # [DIAG] No subsampling needed, but still record R distribution
        _pop_R = reliability_for_sampling.detach()
        _sampling_diag = {
            "population_R_mean": _pop_R.mean().item(),
            "sampled_R_mean": _pop_R.mean().item(),  # same (no subsampling)
            "sampling_bias": 0.0,
            "population_R_lt03": (_pop_R < 0.3).float().mean().item(),
            "sampled_R_lt03": (_pop_R < 0.3).float().mean().item(),
            "population_n": int(_pop_R.numel()),
            "sampled_n": int(_pop_R.numel()),
            "subsampled": False,
        }

    # ---- 1. get ref pixel coords & warp to target ----
    pixels_0 = torch.stack([ix, iy], dim=-1).float().to("cuda")  # (H, W, 2)
    pixels_0_valid = pixels_0.reshape(-1, 2)[valid_indices]       # (M, 2)
    pixels_1_valid = _pixels_warp(H_ref_to_neareast.reshape(-1, 3, 3), pixels_0_valid)

    # normalise to [-1, 1] for grid_sample
    px0_norm = pixels_0_valid.clone()
    px0_norm[:, 0] = 2 * px0_norm[:, 0] / (W - 1) - 1.0
    px0_norm[:, 1] = 2 * px0_norm[:, 1] / (H - 1) - 1.0

    px1_norm = pixels_1_valid.clone()
    px1_norm[:, 0] = 2 * px1_norm[:, 0] / (W - 1) - 1.0
    px1_norm[:, 1] = 2 * px1_norm[:, 1] / (H - 1) - 1.0

    # ---- 2. sample normals from both views ----
    dn0 = render_pkg[normal_key]          # (3, H, W)
    dn1 = nearest_render_pkg[normal_key]  # (3, H, W)

    n0_cam = _sample_normal_map(dn0, px0_norm, H, W)  # (3, M)
    n1_cam = _sample_normal_map(dn1, px1_norm, H, W)  # (3, M)

    # ---- 3. transform camera frame -> world frame ----
    R0 = torch.tensor(viewpoint_cam.R, dtype=torch.float32, device="cuda")
    R1 = torch.tensor(nearest_cam.R, dtype=torch.float32, device="cuda")

    n0_world = (n0_cam.T @ R0.transpose(-1, -2)).T  # (3, M)
    n1_world = (n1_cam.T @ R1.transpose(-1, -2)).T  # (3, M)

    n0_world = F.normalize(n0_world, dim=0)
    n1_world = F.normalize(n1_world, dim=0)

    # ---- 4. cosine similarity (always computed, used for diagnostics) ----
    cos_sim = (n0_world * n1_world).sum(dim=0)  # (M,)

    # ---- 5. outlier mask ----
    if loss_type == "l1":
        # MNE-style: reject pairs with L1 > 0.52 (~30 deg)
        l1_diff = (n0_world - n1_world).abs().sum(0)  # (M,)
        ok_mask = l1_diff < 0.52
    else:
        # Cosine mode: reject near-opposite normals
        ok_mask = cos_sim > -0.5

    if ok_mask.sum() == 0:
        return result

    # ---- 6. loss computation ----
    if loss_type == "l1":
        loss_per_pixel = l1_diff[ok_mask]
        if edge_weight is not None:
            # MNE convention: weight includes outlier masking, squared
            ew = edge_weight.clone()
            ew[~ok_mask] = 0
            ew = ew.clamp(0, 1).detach() ** 2
            # mean over ALL M valid pixels (MNE convention — outliers contribute 0)
            loss = (ew * l1_diff).mean()
        elif reliability_weights is not None:
            r_w = reliability_weights[ok_mask].detach()
            loss = (loss_per_pixel * r_w).sum() / (r_w.sum() + 1e-8)
        else:
            loss = loss_per_pixel.mean()
    else:  # cosine
        loss_per_pixel = 1.0 - cos_sim[ok_mask]
        if reliability_weights is not None:
            r_w = reliability_weights[ok_mask].detach()
            loss = (loss_per_pixel * r_w).sum() / (r_w.sum() + 1e-8)
        else:
            loss = loss_per_pixel.mean()

    # ---- diagnostic weights ----
    if edge_weight is not None:
        _diag_w = edge_weight[ok_mask].clamp(0, 1).detach() ** 2
    elif reliability_weights is not None:
        _diag_w = reliability_weights[ok_mask].detach()
    else:
        _diag_w = None

    result["loss"] = loss
    result["has_valid"] = True
    result["n_valid"] = int(ok_mask.sum().item())
    result["cos_sim"] = cos_sim          # (M,)
    result["ok_mask"] = ok_mask          # (M,) bool
    result["loss_per_pixel"] = loss_per_pixel  # (M_ok,)
    result["weights"] = _diag_w          # (M_ok,) or None
    result["pixel_indices"] = valid_indices    # (M,)
    result["sampling_diag"] = _sampling_diag   # [DIAG] inverse R sampling stats
    return result
