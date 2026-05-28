"""
[DIAG] Diagnostic utilities for reliability innovation debugging.
This entire module is for debugging only and can be safely deleted after diagnosis.

Usage:
    from utils.diagnostics import DiagnosticTracker
    diag = DiagnosticTracker(debug_path, tb_writer)

All diagnostic code in the project is marked with # [DIAG] comments.
To remove all diagnostics:
    grep -rn '\[DIAG\]' TSGS/  # find all diagnostic lines
    rm TSGS/utils/diagnostics.py  # remove this module
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict


class DiagnosticTracker:
    """Tracks and logs intermediate diagnostic statistics for reliability innovations.
    
    Designed to be non-intrusive: all methods are no-ops if disabled.
    """

    def __init__(self, debug_path, tb_writer=None, enabled=True, log_every=100):
        self.enabled = enabled
        self.debug_path = debug_path
        self.tb_writer = tb_writer
        self.log_every = log_every
        self._current = {}  # current iteration diagnostics
        self._jsonl_path = os.path.join(debug_path, "diagnostics.jsonl") if debug_path else None
        # EMA trackers for progress bar display
        self._ema = defaultdict(float)
        self._ema_alpha = 0.1  # slow EMA for stability
    
    def reset_iter(self):
        """Call at the start of each iteration."""
        self._current = {}
    
    # Innovation 2: dual-view reliability diagnostics
    
    def record_innov2(self, d_mask_before, d_mask_after,
                      weights_before_sum, weights_after_sum,
                      dual_rel, r_ref_flat, r_tgt_sampled):
        """Record Innovation 2 diagnostics. Call right after dual_rel is applied."""
        if not self.enabled:
            return
        with torch.no_grad():
            n_before = d_mask_before.sum().item()
            n_after = d_mask_after.sum().item()
            
            # Only compute on originally valid pixels
            valid = d_mask_before
            dr_valid = dual_rel[valid] if valid.any() else dual_rel
            
            self._current.update({
                "innov2/dual_rel_mean": dr_valid.mean().item() if valid.any() else 0.0,
                "innov2/dual_rel_median": dr_valid.median().item() if valid.any() else 0.0,
                "innov2/dual_rel_lt01": (dr_valid < 0.1).float().mean().item() if valid.any() else 0.0,
                "innov2/dual_rel_lt03": (dr_valid < 0.3).float().mean().item() if valid.any() else 0.0,
                "innov2/d_mask_before": float(n_before),
                "innov2/d_mask_after": float(n_after),
                "innov2/d_mask_survival": n_after / max(n_before, 1),
                "innov2/weights_attenuation": weights_after_sum / max(weights_before_sum, 1e-8),
                "innov2/r_ref_mean": r_ref_flat[valid].mean().item() if valid.any() else 0.0,
                "innov2/r_tgt_mean": r_tgt_sampled[valid].mean().item() if valid.any() else 0.0,
            })
    
    # Innovation 3: densification diagnostics
    
    def record_innov3_densify(self, diag_info):
        """Record Innovation 3 diagnostics from densify_and_prune return value."""
        if not self.enabled or not diag_info:
            return
        for k, v in diag_info.items():
            self._current["innov3/{}".format(k)] = float(v)
    
    # R map quality diagnostics
    
    def record_rmap(self, R_map):
        """Record R map quality stats."""
        if not self.enabled or R_map is None:
            return
        with torch.no_grad():
            r = R_map.reshape(-1)
            self._current.update({
                "rmap/R_mean": r.mean().item(),
                "rmap/R_std": r.std().item(),
                "rmap/R_lt01": (r < 0.1).float().mean().item(),
                "rmap/R_lt03": (r < 0.3).float().mean().item(),
                "rmap/R_gt07": (r > 0.7).float().mean().item(),
                "rmap/R_gt09": (r > 0.9).float().mean().item(),
            })
    

    # NCC R-map modulation diagnostics
    
    def record_ncc_modulation(self, ncc_before_R, ncc_after_R, rel_weights):
        """Record NCC loss before/after R-map modulation."""
        if not self.enabled:
            return
        with torch.no_grad():
            rw = rel_weights.detach()
            self._current.update({
                "ncc/R_weight_mean": rw.mean().item(),
                "ncc/R_weight_lt03": (rw < 0.3).float().mean().item(),
                "ncc/R_weight_gt07": (rw > 0.7).float().mean().item(),
                "ncc/ncc_before_R": ncc_before_R,
                "ncc/ncc_after_R": ncc_after_R,
                "ncc/attenuation_ratio": ncc_after_R / max(ncc_before_R, 1e-8),
            })
    
    # MNE inverse reliability sampling diagnostics
    
    def record_mne_sampling(self, mne_diag):
        """Record MNE sampling distribution and loss diagnostics."""
        if not self.enabled or not mne_diag:
            return
        for k, v in mne_diag.items():
            self._current["mne/{}".format(k)] = float(v)

    # Innovation 3: per-Gaussian reliability sampling diagnostics
    
    def record_per_gs_reliability(self, per_gs_rel):
        """Record per-Gaussian reliability from R-map grid_sample. Call each iteration when available."""
        if not self.enabled:
            return
        with torch.no_grad():
            r = per_gs_rel
            valid = r > 0  # Gaussians projecting inside image
            rv = r[valid] if valid.any() else r
            self._current.update({
                "prune/per_gs_R_mean": rv.mean().item(),
                "prune/per_gs_R_median": rv.median().item(),
                "prune/per_gs_R_lt01": (rv < 0.1).float().mean().item(),
                "prune/per_gs_R_lt03": (rv < 0.3).float().mean().item(),
                "prune/per_gs_R_gt07": (rv > 0.7).float().mean().item(),
                "prune/per_gs_n_valid": valid.sum().item(),
                "prune/per_gs_n_total": r.shape[0],
            })
    
    # Multi-view trim diagnostics
    
    def record_mv_trim(self, trimmed_count, total_after):
        """Record multi-view observe trim results."""
        if not self.enabled:
            return
        self._current.update({
            "prune/mv_trim_count": float(trimmed_count),
            "prune/mv_trim_total_after": float(total_after),
        })
    
    # Flush: write to TB and JSONL
    
    def flush(self, iteration):
        """Write accumulated diagnostics to TensorBoard and JSONL file."""
        if not self.enabled or not self._current:
            return
        
        # Update EMA
        for k, v in self._current.items():
            self._ema[k] = self._ema_alpha * v + (1 - self._ema_alpha) * self._ema[k]
        
        # Write to TensorBoard
        if self.tb_writer and iteration % self.log_every == 0:
            for k, v in self._current.items():
                self.tb_writer.add_scalar("diag/{}".format(k), v, iteration)
        
        # Write to JSONL (less frequent to avoid I/O overhead)
        if self._jsonl_path and iteration % 500 == 0:
            record = {"iteration": iteration}
            record.update(self._current)
            try:
                with open(self._jsonl_path, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass  # never crash training for diagnostics
    
    def get_summary_str(self):
        """Return a short summary string for progress bar (optional)."""
        if not self.enabled:
            return ""
        parts = []
        for key in ["innov2/dual_rel_mean", "innov2/d_mask_survival", 
                     "innov2/weights_attenuation", "rmap/R_mean"]:
            if key in self._ema:
                short = key.split("/")[-1][:8]
                parts.append("{}={:.3f}".format(short, self._ema[key]))
        return " ".join(parts)
