import torch
import torch.distributed as dist
import matplotlib
# Set backend to Agg before importing pyplot to avoid GUI errors on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import io
import numpy as np

class PKMMonitor:
    _counts = None # [n_groups, n_keys]
    _n_groups = 5
    _n_keys = 128
    _device = None
    _enabled = False

    @classmethod
    def init(cls, n_groups=5, n_keys=128, device='cuda'):
        cls._n_groups = n_groups
        cls._n_keys = n_keys
        cls._device = device
        cls._enabled = True
        cls.reset()

    @classmethod
    def reset(cls):
        if cls._device:
             cls._counts = torch.zeros((cls._n_groups, cls._n_keys), dtype=torch.float32, device=cls._device)

    @classmethod
    def update(cls, group_ids, indices, batch_size, seq_len, scores=None):
        """
        group_ids: [Batch]
        indices: [Batch*SeqLen, Heads, TopK]
        scores:  [Batch*SeqLen, Heads, TopK] (optional)
        """
        if not cls._enabled or cls._counts is None:
            return

        # Ensure devices match
        if group_ids.device != cls._counts.device:
            group_ids = group_ids.to(cls._counts.device)
        
        # indices might be on different device if model is on different GPU than monitor? 
        # Usually same.
        
        # 1. Reshape indices: [B, T, H, K]
        # Note: indices contains KeyIDs (0 to 127)
        B, T = batch_size, seq_len
        if indices.shape[0] != B * T:
            # Mismatch, maybe due to padding or implementation detail
            return 
            
        indices = indices.view(B, T, -1) # [B, T, H*K]
        if scores is not None:
            if scores.device != cls._counts.device:
                scores = scores.to(cls._counts.device)
            if scores.shape[0] != B * T:
                scores = None
            else:
                scores = scores.view(B, T, -1)
        
        # 2. Expand group_ids: [B] -> [B, T, H*K]
        # Only valid groups (>=0)
        valid_mask = (group_ids >= 0)
        if not valid_mask.any():
            return
            
        # Filter batch
        group_ids = group_ids[valid_mask] # [B_valid]
        indices = indices[valid_mask]     # [B_valid, T, H*K]
        
        # Expand groups
        # [B_valid, 1, 1] -> [B_valid, T, H*K]
        groups_expanded = group_ids.view(-1, 1, 1).expand_as(indices)
        
        g_flat = groups_expanded.reshape(-1)
        i_flat = indices.reshape(-1)

        if scores is not None:
            s_flat = scores[valid_mask].reshape(-1)
            keep = torch.isfinite(s_flat) & (s_flat > -1e8)
            if not keep.any():
                return
            g_flat = g_flat[keep]
            i_flat = i_flat[keep]
        
        # 4. Accumulate
        # We want counts[g, i] += 1
        # Use flat index: g * n_keys + i
        flat_indices = g_flat * cls._n_keys + i_flat.long()
        
        # Count occurrences
        # bincount is fast for 1D
        # max value = (n_groups-1)*n_keys + (n_keys-1) = n_groups*n_keys - 1
        
        # Ensure flat_indices are within range
        mask = (flat_indices >= 0) & (flat_indices < cls._n_groups * cls._n_keys)
        flat_indices = flat_indices[mask]
        
        if flat_indices.numel() > 0:
            updates = torch.bincount(flat_indices, minlength=cls._n_groups * cls._n_keys).float()
            cls._counts.view(-1).add_(updates)

    @classmethod
    def get_and_reset(cls):
        if cls._counts is None:
            return None
            
        # Distributed Sync
        if dist.is_initialized():
            dist.all_reduce(cls._counts, op=dist.ReduceOp.SUM)
            
        # Get data to CPU
        data = cls._counts.cpu().numpy()
        
        # Reset
        cls._counts.zero_()
        
        return data

    @classmethod
    def plot_heatmap(cls, data):
        fig, ax = plt.subplots(figsize=(10, 4))

        row_sums = data.sum(axis=1, keepdims=True) + 1e-9
        norm_data = data / row_sums

        nz = norm_data[norm_data > 0]
        if nz.size:
            vmax = float(np.quantile(nz, 0.99))
            vmax = max(vmax, float(nz.max()))
            vmax = min(vmax, 1.0)
            vmax = max(vmax, 1e-6)
        else:
            vmax = 1.0

        im = ax.imshow(
            norm_data,
            aspect="auto",
            cmap="viridis",
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax),
            interpolation="nearest",
        )
        ax.set_xlabel("Memory Keys (0-127)")
        ax.set_ylabel("User Groups (0-4)")
        ax.set_title(f"Memory Activation Heatmap (Normalized per Group, vmax={vmax:.4g})")
        plt.colorbar(im, ax=ax)

        n_keys = data.shape[1]
        for i in range(1, 5):
            x = (i * n_keys / 5)
            ax.axvline(x=x, color="red", linestyle="--", alpha=0.5)

        return fig