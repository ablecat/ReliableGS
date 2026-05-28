"""
Multi-view geometric statistics reliability module (Gradient Consensus).

Computes per-view, per-pixel reliability maps R_i(u,v) by measuring
depth reprojection consistency AND normal consistency across K nearest
neighbor views.

    R_i(u,v) = exp( -Var_j(e_d) / tau_v  -  (1 - ||mean_n||) / tau_c )

where:
    Var_j(e_d)        = variance of depth reprojection errors across K neighbors
    ||mean_n||        = resultant length of K neighbor normal vectors
                        (1 = all agree, 0 = fully random)

Design decisions (confirmed):
    K              = 5        # number of neighbor views
    update_every   = 500      # recompute buffer every N training iterations
    start_iter     = 3000     # first iteration to start computing stats
    use_from_iter  = 7000     # first iteration to apply weights in losses
    tau_v          = 0.5      # temperature for depth variance term
    tau_c          = 0.5      # temperature for normal consensus term
    dtype          = float32
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Main update routine
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_reliability_buffer(
    scene,
    gaussians,
    pipe,
    background,
    render_fn,
    app_model,
    reliability_buffer: dict,
    K: int = 5,
    tau_d: float = 0.5,
    tau_n: float = 0.5,
    pixel_noise_th: float = 30.0,
    save_dir: str = None,
    iteration: int = 0,
    n_diag_samples: int = 4,
    adaptive_tau: bool = False,
    tau_quantile: float = 0.75,
    tau_alpha: float = 1.0,
    tau_d_floor: float = 0.1,
    tau_n_floor: float = 0.05,
):
    """Recompute per-view reliability maps for all training cameras.

    For every training camera *i* the function:
      1. Renders depth + normal for camera *i* and its K neighbours.
      2. Computes depth-based reprojection pixel noise per neighbour.
      3. Computes neighbour normal vectors in world frame.
      4. Computes VARIANCE of depth errors and CONSENSUS of normals.
      5. Stores  R = exp(-var_depth/tau_v - (1-||mean_n||)/tau_c).

    The result is cached in *reliability_buffer* (uid -> (H,W) float32 GPU
    tensor) and can be queried via get_reliability_weights().
    """
    train_cameras = scene.getTrainCameras()
    n_cams = len(train_cameras)
    logger.info(
        f"[Reliability] Updating buffer (consensus) for {n_cams} cams "
        f"(K={K}, tau_v={tau_d}, tau_c={tau_n}, depth=plane_depth)"
    )

    # ------------------------------------------------------------------
    # Phase 1: render depth & rendered_normal for every training camera
    # ------------------------------------------------------------------
    depth_cache = {}   # uid -> (1, H, W)  GPU float32  (plane_depth)
    normal_cache = {}  # uid -> (3, H, W) GPU float32  (camera-frame rendered normals)

    for cam in train_cameras:
        pkg = render_fn(
            cam, gaussians, pipe, background,
            app_model=app_model,
            return_plane=True,
            return_depth_normal=False,
        )
        depth_cache[cam.uid] = pkg["plane_depth"].detach()       # (1,H,W)
        normal_cache[cam.uid] = pkg["rendered_normal"].detach()  # (3,H,W)
        del pkg
        torch.cuda.empty_cache()

    # Select sample cameras for diagnostic visualization
    _diag_err_maps = {}
    _diag_sample_indices = set()
    _cam_mean_data = {}  # uid -> (mean_depth, mean_normal, has_valid, H, W)
    if save_dir and n_diag_samples > 0 and n_cams > 0:
        _step = max(1, n_cams // n_diag_samples)
        _diag_sample_indices = set(range(0, n_cams, _step))

    # ------------------------------------------------------------------
    # Phase 2: per-camera reliability computation
    # ------------------------------------------------------------------
    updated = 0
    for _cam_idx, cam_i in enumerate(train_cameras):
        if len(cam_i.nearest_id) == 0:
            continue

        H_i = cam_i.image_height
        W_i = cam_i.image_width
        device = depth_cache[cam_i.uid].device

        depth_i_raw = depth_cache[cam_i.uid]                    # (1, H, W)
        depth_i = depth_i_raw.squeeze()                          # (H, W)
        normal_i = normal_cache[cam_i.uid]                      # (3, H, W) cam frame

        # Unproject cam_i depth -> world-space 3D points  (H*W, 3)
        pts_world = gaussians.get_points_from_depth(cam_i, depth_i)

        # Build pixel grid once
        ix = torch.arange(W_i, device=device, dtype=torch.float32)
        iy = torch.arange(H_i, device=device, dtype=torch.float32)
        gx, gy = torch.meshgrid(ix, iy, indexing="xy")
        pixels_flat = torch.stack([gx.reshape(-1),
                                   gy.reshape(-1)], dim=-1)      # (H*W, 2)

        # cam_i rotation matrix (for normal -> world)
        R_i = torch.tensor(cam_i.R, dtype=torch.float32, device=device)  # (3,3)

        # Normal of cam_i in world frame  (3, H*W)
        n_i_cam = normal_i.reshape(3, -1)                        # (3, H*W)
        n_i_world = R_i.T @ n_i_cam                              # (3, H*W)
        n_i_world = F.normalize(n_i_world, dim=0)                # unit

        # Select K neighbours
        nb_indices = cam_i.nearest_id[:K]

        # Accumulators (consensus R-map)
        noise_sum      = torch.zeros(H_i * W_i, device=device)       # sum pn
        noise_sum_sq   = torch.zeros(H_i * W_i, device=device)       # sum pn^2
        normal_vec_sum = torch.zeros(3, H_i * W_i, device=device)    # sum n_j_world
        valid_cnt      = torch.zeros(H_i * W_i, device=device)

        for nb_idx in nb_indices:
            cam_j = train_cameras[nb_idx]
            depth_j_raw = depth_cache[cam_j.uid]
            depth_j = depth_j_raw.squeeze()
            normal_j = normal_cache[cam_j.uid]                  # (3, H_j, W_j)

            # ---- depth reprojection error ----
            pts_in_j = (pts_world @ cam_j.world_view_transform[:3, :3]
                        + cam_j.world_view_transform[3, :3])
            map_z, d_mask = gaussians.get_points_depth_in_depth_map(
                cam_j, depth_j_raw, pts_in_j)

            pts_in_j_norm   = pts_in_j / (pts_in_j[:, 2:3] + 1e-8)
            pts_in_j_scaled = pts_in_j_norm * map_z.squeeze()[..., None]

            R_j = torch.tensor(cam_j.R, dtype=torch.float32, device=device)
            T_j = torch.tensor(cam_j.T, dtype=torch.float32, device=device)
            pts_reproj_world = (pts_in_j_scaled - T_j) @ R_j.T

            pts_in_i = (pts_reproj_world @ cam_i.world_view_transform[:3, :3]
                        + cam_i.world_view_transform[3, :3])
            px = pts_in_i[:, 0] * cam_i.Fx / (pts_in_i[:, 2] + 1e-8) + cam_i.Cx
            py = pts_in_i[:, 1] * cam_i.Fy / (pts_in_i[:, 2] + 1e-8) + cam_i.Cy
            reproj = torch.stack([px, py], dim=-1)
            pn = torch.norm(reproj - pixels_flat, dim=-1).clamp(max=pixel_noise_th)

            # ---- normal disagreement ----
            # Project cam_i pixel positions into cam_j image to sample normals
            proj_x = pts_in_j[:, 0] * cam_j.Fx / (pts_in_j[:, 2] + 1e-8) + cam_j.Cx
            proj_y = pts_in_j[:, 1] * cam_j.Fy / (pts_in_j[:, 2] + 1e-8) + cam_j.Cy
            H_j = cam_j.image_height
            W_j = cam_j.image_width
            # Normalise to [-1, 1] for grid_sample
            gx_n = 2.0 * proj_x / (W_j - 1) - 1.0
            gy_n = 2.0 * proj_y / (H_j - 1) - 1.0
            grid_pts = torch.stack([gx_n, gy_n], dim=-1).reshape(1, 1, -1, 2)
            # Sample cam_j rendered_normal
            n_j_cam_sampled = F.grid_sample(
                normal_j.unsqueeze(0), grid_pts,
                mode="bilinear", padding_mode="zeros", align_corners=True
            ).squeeze()                                          # (3, H_i*W_i)
            n_j_cam_sampled = F.normalize(n_j_cam_sampled, dim=0)

            # Transform cam_j normal to world frame
            R_j_mat = torch.tensor(cam_j.R, dtype=torch.float32, device=device)
            n_j_world = R_j_mat.T @ n_j_cam_sampled
            n_j_world = F.normalize(n_j_world, dim=0)

            # Accumulate only valid pixels (consensus: vector sum)
            dm = d_mask.reshape(-1)
            noise_sum[dm]      += pn[dm]
            noise_sum_sq[dm]   += pn[dm] ** 2
            normal_vec_sum[:, dm] += n_j_world[:, dm]
            valid_cnt[dm]      += 1.0

        # ---- combine into consensus reliability map ----
        has_valid = valid_cnt > 0
        # Depth: variance of reprojection errors  Var = E[x^2] - E[x]^2
        mean_pn = torch.zeros_like(noise_sum)
        mean_sq = torch.zeros_like(noise_sum)
        mean_pn[has_valid] = noise_sum[has_valid]    / valid_cnt[has_valid]
        mean_sq[has_valid] = noise_sum_sq[has_valid] / valid_cnt[has_valid]
        var_depth = (mean_sq - mean_pn ** 2).clamp(min=0.0)
        # Normal: dissensus = 1 - resultant_length
        mean_nvec = torch.zeros_like(normal_vec_sum)
        mean_nvec[:, has_valid] = normal_vec_sum[:, has_valid] / valid_cnt[has_valid]
        resultant_len = mean_nvec.norm(dim=0)        # (H*W,)  in [0, 1]
        dissensus = (1.0 - resultant_len).clamp(min=0.0)

        # Store error maps for diagnostic samples
        if _cam_idx in _diag_sample_indices:
            # On-demand depth_normal for diagnostic (avoid caching all 350)
            _dn_pkg = render_fn(
                cam_i, gaussians, pipe, background,
                app_model=app_model,
                return_plane=True,
                return_depth_normal=True,
            )
            _dn = _dn_pkg.get("depth_normal", torch.zeros(3, H_i, W_i, device=device)).detach()
            del _dn_pkg
            _diag_err_maps[cam_i.uid] = {
                "depth_err": var_depth.reshape(H_i, W_i).clone(),
                "normal_err": dissensus.reshape(H_i, W_i).clone(),
                "plane_depth": depth_cache[cam_i.uid].squeeze().clone(),
                "rendered_normal": normal_cache[cam_i.uid].clone(),
                "depth_normal": _dn.clone(),
            }
            del _dn

        # Store intermediate results for adaptive tau (two-pass)
        _cam_mean_data[cam_i.uid] = (var_depth, dissensus, has_valid, H_i, W_i)
        updated += 1

    # ── Adaptive tau: compute from error distribution ──
    if adaptive_tau and _cam_mean_data:
        all_d = torch.cat([vd[hv] for vd, ds, hv, h, w in _cam_mean_data.values() if hv.any()])
        all_n = torch.cat([ds[hv] for vd, ds, hv, h, w in _cam_mean_data.values() if hv.any()])
        if all_d.numel() > 0 and all_n.numel() > 0:
            # Subsample to avoid torch.quantile OOM on large tensors
            _max_samples = 1_000_000
            if all_d.numel() > _max_samples:
                _idx = torch.randint(0, all_d.numel(), (_max_samples,), device=all_d.device)
                all_d = all_d[_idx]
                all_n = all_n[_idx]
            _med_d = torch.median(all_d).item()
            _med_n = torch.median(all_n).item()
            _raw_qd = torch.quantile(all_d, tau_quantile).item()
            _raw_qn = torch.quantile(all_n, tau_quantile).item()
            _p95_d = torch.quantile(all_d, 0.95).item()
            _p95_n = torch.quantile(all_n, 0.95).item()

            # ── Contribution-balanced adaptive tau ──
            # Goal: median(e_d)/tau_d == median(e_n)/tau_n so both terms
            # contribute equally for the "typical" pixel.
            # Step 1: set tau_d from quantile, apply floor FIRST
            tau_d_raw = _raw_qd * tau_alpha
            tau_n_raw = _raw_qn * tau_alpha
            tau_d = max(tau_d_raw, tau_d_floor)
            # Step 2: balance tau_n to match tau_d's ACTUAL contribution
            # (after floor clamp, not before)
            if _med_d > 0 and _med_n > 0:
                # tau_n = med_n * tau_d / med_d  →  med_d/tau_d == med_n/tau_n
                tau_n_balanced = _med_n * tau_d / (_med_d + 1e-10)
                tau_n = max(tau_n_balanced, tau_n_floor)
            else:
                tau_n = max(tau_n_raw, tau_n_floor)
            # Verify balanced contribution
            _cd = _med_d / (tau_d + 1e-10)
            _cn = _med_n / (tau_n + 1e-10)
            logger.info(
                f"[Reliability] Consensus stats: "
                f"var_depth median={_med_d:.4f} q75={_raw_qd:.4f} p95={_p95_d:.4f} | "
                f"dissensus median={_med_n:.6f} q75={_raw_qn:.6f} p95={_p95_n:.6f}"
            )
            logger.info(
                f"[Reliability] Balanced tau: tau_v={tau_d:.4f} tau_c={tau_n:.6f} | "
                f"contrib var={_cd:.3f} dissensus={_cn:.3f} ratio={_cd/(_cn+1e-10):.2f}"
            )

    # Pack balanced tau diagnostics for panel display
    _bal_info = {}
    if adaptive_tau and '_med_d' in dir():
        _bal_info = dict(
            med_d=_med_d, med_n=_med_n, q75_d=_raw_qd, q75_n=_raw_qn,
            p95_d=_p95_d, p95_n=_p95_n,
            tau_d_raw=tau_d_raw, tau_n_raw=tau_n_raw,
            tau_n_balanced=locals().get('tau_n_balanced', tau_n),
            contrib_d=_cd, contrib_n=_cn,
        )

    # ── Second pass: compute R with (possibly adapted) tau ──
    for uid, (var_depth, dissensus, has_valid, H_i, W_i) in _cam_mean_data.items():
        R = torch.zeros_like(var_depth)
        R[has_valid] = torch.exp(
            -var_depth[has_valid] / tau_d - dissensus[has_valid] / tau_n
        )
        reliability_buffer[uid] = R.reshape(H_i, W_i).float()

    # ── Save diagnostic panel ──
    if save_dir and _diag_err_maps:
        try:
            save_reliability_diagnostic(
                reliability_buffer, train_cameras, _diag_err_maps,
                iteration, save_dir, tau_d, tau_n, bal_info=_bal_info,
                )
        except Exception as e:
            logger.warning(f"[Reliability] Diagnostic save failed: {e}")

    # ------------------------------------------------------------------
    # Phase 3: release caches
    # ------------------------------------------------------------------
    del depth_cache, normal_cache
    torch.cuda.empty_cache()

    # quick stats
    if updated > 0:
        vals = [v.mean().item() for v in reliability_buffer.values()]
        avg_r = sum(vals) / len(vals)
        min_r = min(vals)
        max_r = max(vals)
        logger.info(
            f"[Reliability] Done.  updated={updated}/{n_cams}  "
            f"mean_R={avg_r:.4f}  min_R={min_r:.4f}  max_R={max_r:.4f}"
        )


# ---------------------------------------------------------------------------
#  Query helpers
# ---------------------------------------------------------------------------

def get_reliability_weights(
    reliability_buffer: dict,
    viewpoint_cam,
    valid_indices: torch.Tensor,
) -> torch.Tensor:
    """Look up reliability for a set of valid pixel indices.

    Args:
        reliability_buffer:  uid -> (H, W) float32 tensor.
        viewpoint_cam:       Current camera.
        valid_indices:       (M,) int64 flat pixel indices in [0, H*W).

    Returns:
        (M,) float32 tensor in [0, 1].  Ones if no data for this view.
    """
    uid = viewpoint_cam.uid
    if uid not in reliability_buffer:
        return torch.ones(valid_indices.shape[0],
                          device=valid_indices.device, dtype=torch.float32)
    R_flat = reliability_buffer[uid].reshape(-1).to(valid_indices.device)
    return R_flat[valid_indices]


def get_reliability_map(
    reliability_buffer: dict,
    viewpoint_cam,
) -> torch.Tensor:
    """Return full (H, W) reliability map.  None if unavailable."""
    uid = viewpoint_cam.uid
    return reliability_buffer.get(uid, None)


# ---------------------------------------------------------------------------
#  R-map Diagnostic Visualization
# ---------------------------------------------------------------------------

def _render_r_histogram(all_R, h, w):
    """Render R-value histogram as a BGR image using pure numpy/cv2."""
    img = np.full((h, w, 3), 250, dtype=np.uint8)  # light background

    n_bins = 50
    counts, edges = np.histogram(all_R, bins=n_bins, range=(0.0, 1.0))
    max_count = max(counts.max(), 1)

    ml, mr, mt, mb = 45, 10, 25, 30  # margins
    pw = w - ml - mr
    ph = h - mt - mb
    bar_w = max(pw // n_bins, 1)

    for i, c in enumerate(counts):
        bar_h = int(c / max_count * ph)
        x1 = ml + i * bar_w
        x2 = x1 + bar_w - 1
        y2 = h - mb
        y1 = y2 - bar_h
        val = (edges[i] + edges[i + 1]) / 2
        color = cv2.applyColorMap(
            np.array([[int(val * 255)]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS
        )[0, 0].tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "R-value Distribution", (ml, 16), font, 0.45,
                (30, 30, 30), 1, cv2.LINE_AA)
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tx = ml + int(tick * (pw - bar_w))
        cv2.putText(img, f"{tick:.2f}", (tx - 8, h - 8), font, 0.3,
                    (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(img, (tx, h - mb), (tx, h - mb + 4), (160, 160, 160), 1)

    # Y-axis label
    cv2.putText(img, f"{max_count}", (2, mt + 10), font, 0.3,
                (80, 80, 80), 1, cv2.LINE_AA)
    return img


def _render_stats_panel(all_R, h, w, iteration, tau_d, tau_n, n_cams,
                        bal_info=None):
    """Render statistics text as a BGR image."""
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = [
        f"Iteration: {iteration}   Views: {n_cams}",
        f"tau_v = {tau_d:.4f}   tau_c = {tau_n:.6f}",
        "",
        f"R  mean={all_R.mean():.4f}  std={all_R.std():.4f}  med={np.median(all_R):.4f}",
        f"R <0.1:{(all_R<0.1).mean()*100:.1f}%  <0.3:{(all_R<0.3).mean()*100:.1f}%"
        f"  >0.7:{(all_R>0.7).mean()*100:.1f}%  >0.9:{(all_R>0.9).mean()*100:.1f}%",
    ]

    if bal_info:
        b = bal_info
        lines += [
            "",
            "--- Balanced Tau (Consensus) ---",
            f"var_depth : med={b['med_d']:.4f}  q75={b['q75_d']:.4f}  p95={b['p95_d']:.4f}",
            f"dissensus: med={b['med_n']:.6f} q75={b['q75_n']:.6f} p95={b['p95_n']:.6f}",
            f"tau_v_raw={b['tau_d_raw']:.4f}  tau_c_raw={b['tau_n_raw']:.6f}",
            f"tau_c_balanced={b['tau_n_balanced']:.6f}",
            f"contrib: var={b['contrib_d']:.3f}  dissensus={b['contrib_n']:.3f}"
            f"  ratio={b['contrib_d']/(b['contrib_n']+1e-10):.2f}",
        ]

    y = 16
    for line in lines:
        if line == "":
            y += 6
            continue
        cv2.putText(img, line, (8, y), font, 0.32, (30, 30, 30), 1,
                    cv2.LINE_AA)
        y += 14

    return img


@torch.no_grad()
def save_reliability_diagnostic(
    reliability_buffer: dict,
    train_cameras,
    sample_err_maps: dict,
    iteration: int,
    save_dir: str,
    tau_d: float,
    tau_n: float,
    bal_info: dict = None,
):
    """Save a composite diagnostic image for R-map quality inspection.

    Layout (per sample view row):
        | GT | R-map (JET) | PlaneDepth (JET) | AltDepth (JET) | VarDepth (JET) | Dissensus (JET) |
    Bottom row:
        | R-value Histogram | Global Statistics |
    """
    rmap_dir = os.path.join(save_dir, "rmap_diag")
    os.makedirs(rmap_dir, exist_ok=True)

    cell_w = 600
    sample_uids = list(sample_err_maps.keys())
    if not sample_uids:
        return

    cam_by_uid = {c.uid: c for c in train_cameras}

    # Compute cell_h from actual camera aspect ratio
    _first_cam = cam_by_uid.get(sample_uids[0])
    if _first_cam is not None:
        cell_h = max(100, int(cell_w * _first_cam.image_height / _first_cam.image_width))
    else:
        cell_h = 200

    n_cols = 7  # GT | R-map | PlaneDepth | RenderNormal | DepthNormal | DepthErr | NormalErr

    # ── Global normalization for error maps ──
    all_d_vals = torch.cat(
        [sample_err_maps[u]["depth_err"].flatten() for u in sample_uids])
    all_n_vals = torch.cat(
        [sample_err_maps[u]["normal_err"].flatten() for u in sample_uids])
    # Use 99-th percentile to avoid outlier domination
    pos_d = all_d_vals[all_d_vals > 0]
    max_d = (torch.quantile(pos_d, 0.99).item() * 1.1
             if pos_d.numel() > 0 else 1.0)
    pos_n = all_n_vals[all_n_vals > 0]
    max_n = (torch.quantile(pos_n, 0.99).item() * 1.1
             if pos_n.numel() > 0 else 1.0)

    # ── Per-type normalization for depth visualization ──
    all_pd = torch.cat(
        [sample_err_maps[u]["plane_depth"].flatten() for u in sample_uids
         if "plane_depth" in sample_err_maps[u]])
    pos_pd = all_pd[all_pd > 0]
    if pos_pd.numel() > 0:
        pd_vmin = torch.quantile(pos_pd, 0.02).item()
        pd_vmax = torch.quantile(pos_pd, 0.98).item()
    else:
        pd_vmin, pd_vmax = 0.0, 1.0


    rows = []
    for uid in sample_uids:
        cam = cam_by_uid.get(uid)
        if cam is None:
            continue

        # ---- GT image ----
        try:
            gt = cam.original_image  # (3, H, W) float [0,1]
            if gt is not None:
                gt_np = (gt.cpu().permute(1, 2, 0).numpy()[:, :, ::-1] * 255
                         ).clip(0, 255).astype(np.uint8)
            else:
                gt_np = np.zeros((cam.image_height, cam.image_width, 3),
                                 dtype=np.uint8)
        except Exception:
            gt_np = np.full((cell_h, cell_w, 3), 220, dtype=np.uint8)
        gt_np = cv2.resize(gt_np, (cell_w, cell_h))

        # ---- R-map ----
        R_map = reliability_buffer.get(uid)
        if R_map is not None:
            r_np = (R_map.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            r_color = cv2.applyColorMap(
                cv2.resize(r_np, (cell_w, cell_h)), cv2.COLORMAP_INFERNO)
        else:
            r_color = np.full((cell_h, cell_w, 3), 220, dtype=np.uint8)

        # ---- Depth error ----
        d_err = sample_err_maps[uid]["depth_err"].cpu().numpy()
        d_norm = (d_err / max_d * 255).clip(0, 255).astype(np.uint8)
        d_color = cv2.applyColorMap(
            cv2.resize(d_norm, (cell_w, cell_h)), cv2.COLORMAP_HOT)

        # ---- Plane depth (own range) ----
        pd_raw = sample_err_maps[uid].get("plane_depth")
        if pd_raw is not None:
            pd_np = pd_raw.cpu().numpy()
            pd_norm = ((pd_np - pd_vmin) / (pd_vmax - pd_vmin + 1e-8) * 255
                       ).clip(0, 255).astype(np.uint8)
            pd_color = cv2.applyColorMap(
                cv2.resize(pd_norm, (cell_w, cell_h)), cv2.COLORMAP_MAGMA)
        else:
            pd_color = np.full((cell_h, cell_w, 3), 220, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # ---- Rendered normal ----
        rn_data = sample_err_maps[uid].get("rendered_normal")
        if rn_data is not None:
            rn_np = ((rn_data.cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255
                     ).clip(0, 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
            rn_color = cv2.resize(rn_np, (cell_w, cell_h))
        else:
            rn_color = np.full((cell_h, cell_w, 3), 220, dtype=np.uint8)
        cv2.putText(rn_color, "RenderNormal", (4, 14), font, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # ---- Depth normal ----
        dn_data = sample_err_maps[uid].get("depth_normal")
        if dn_data is not None:
            dn_np = ((dn_data.cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255
                     ).clip(0, 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
            dn_color = cv2.resize(dn_np, (cell_w, cell_h))
        else:
            dn_color = np.full((cell_h, cell_w, 3), 220, dtype=np.uint8)
        cv2.putText(dn_color, "DepthNormal", (4, 14), font, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # ---- Normal error ----
        n_err = sample_err_maps[uid]["normal_err"].cpu().numpy()
        n_norm = (n_err / max_n * 255).clip(0, 255).astype(np.uint8)
        n_color = cv2.applyColorMap(
            cv2.resize(n_norm, (cell_w, cell_h)), cv2.COLORMAP_CIVIDIS)

        # ---- Labels ----
        name = getattr(cam, "image_name", str(uid))
        cv2.putText(gt_np, f"GT: {name}", (4, 14), font, 0.4,
                    (30, 30, 30), 1, cv2.LINE_AA)
        if R_map is not None:
            rm = R_map.mean().item()
            rl = (R_map < 0.3).float().mean().item() * 100
            cv2.putText(r_color, f"R mean={rm:.3f} <0.3:{rl:.0f}%",
                        (4, 14), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pd_color, f"PlaneDepth [{pd_vmin:.2f},{pd_vmax:.2f}]",
                    (4, 14), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(d_color, f"VarDepth (max={max_d:.2f})",
                    (4, 14), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(n_color, f"Dissensus (max={max_n:.3f})",
                    (4, 14), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        row = np.concatenate([gt_np, r_color, pd_color, rn_color, dn_color, d_color, n_color], axis=1)
        rows.append(row)

    if not rows:
        return

    # ── Bottom row: histogram + stats ──
    total_w = cell_w * n_cols
    hist_w = total_w // 2
    stats_w = total_w - hist_w

    all_R_vals = []
    for v in reliability_buffer.values():
        all_R_vals.append(v.cpu().numpy().ravel())
    all_R = np.concatenate(all_R_vals) if all_R_vals else np.array([0.0])

    hist_img = _render_r_histogram(all_R, cell_h, hist_w)
    stats_img = _render_stats_panel(
        all_R, cell_h, stats_w, iteration, tau_d, tau_n, len(train_cameras),
        bal_info=bal_info)
    bottom = np.concatenate([hist_img, stats_img], axis=1)

    panel = np.concatenate(rows + [bottom], axis=0)
    out_path = os.path.join(rmap_dir, f"rmap_diag_{iteration:06d}.png")
    cv2.imwrite(out_path, panel)
    logger.info(f"[Reliability] Saved diagnostic panel: {out_path}")
