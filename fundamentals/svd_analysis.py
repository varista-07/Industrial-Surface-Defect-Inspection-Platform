"""
fundamentals/svd_analysis.py
Singular Value Decomposition (SVD) on NEU-DET surface defect images:
1. SVD factorization: A = U * Sigma * V^T.
2. Singular value spectrum and cumulative energy analysis.
3. Low-rank image reconstruction across ranks k in [1, 5, 20, 50, 100].
4. Quantitative quality metrics: Frobenius error, Compression Ratio, and PSNR (dB).
5. Generates outputs/figures/svd_compression_analysis.png.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compute_psnr(orig, recon):
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels."""
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20.0 * np.log10(max_pixel / np.sqrt(mse))

def run_svd_analysis():
    print("=" * 60)
    print("SVD ANALYSIS & LOW-RANK RECONSTRUCTION ON DEFECT IMAGES")
    print("=" * 60)
    
    # Read sample image from dataset
    config_file = os.path.join("config", "dataset_path.txt")
    with open(config_file, "r") as f:
        rel_path = f.read().strip()
    dataset_root = os.path.abspath(rel_path)
    
    # Select a high-detail defect image (e.g., crazing or scratches)
    sample_path = os.path.join(dataset_root, "IMAGES", "crazing_1.jpg")
    if not os.path.exists(sample_path):
        # Fallback to any available jpg
        sample_path = glob.glob(os.path.join(dataset_root, "IMAGES", "*.jpg"))[0]
        
    img = Image.open(sample_path).convert("L")  # Grayscale (200, 200)
    A = np.array(img, dtype=np.float64)
    H, W = A.shape
    print(f"Loaded image: {os.path.basename(sample_path)}, Shape: {H}x{W}")
    
    # Full SVD: A = U @ np.diag(S) @ Vt
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"Computed SVD: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
    
    # Variance / energy explained by top singular values
    energy = S ** 2
    cum_energy = np.cumsum(energy) / np.sum(energy) * 100.0
    
    print("\nCumulative Energy Explained:")
    ranks_to_check = [1, 5, 20, 50, 100, 150]
    for r in ranks_to_check:
        print(f"  Rank {r:3d}: {cum_energy[r-1]:6.2f}% of total spectral energy")
        
    # Low-rank reconstructions
    eval_ranks = [1, 5, 20, 50, 100]
    reconstructions = {}
    metrics = []
    
    for k in eval_ranks:
        # Reconstruct: U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
        A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        A_k_clipped = np.clip(A_k, 0, 255)
        reconstructions[k] = A_k_clipped
        
        frob_error = np.linalg.norm(A - A_k) / np.linalg.norm(A)
        psnr_val = compute_psnr(A, A_k_clipped)
        # Parameters stored: k * (H + W + 1) vs full H * W
        compressed_params = k * (H + W + 1)
        original_params = H * W
        comp_ratio = original_params / compressed_params
        
        metrics.append({
            "rank": k,
            "frob_error": frob_error,
            "psnr": psnr_val,
            "comp_ratio": comp_ratio,
            "energy_pct": cum_energy[k-1]
        })
        print(f"Rank {k:3d} -> Frobenius Rel Error: {frob_error:.4f}, PSNR: {psnr_val:5.2f} dB, Compression: {comp_ratio:.2f}x")
        
    # Generate comprehensive multi-panel figure
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    fig = plt.figure(figsize=(15, 8))
    
    # Panel 1: Original Image
    ax0 = plt.subplot2grid((2, 6), (0, 0))
    ax0.imshow(A, cmap="gray")
    ax0.set_title("Original (Rank 200)")
    ax0.axis("off")
    
    # Panels 2-6: Reconstructions at ranks [1, 5, 20, 50, 100]
    for idx, k in enumerate(eval_ranks):
        ax = plt.subplot2grid((2, 6), (0, idx + 1))
        ax.imshow(reconstructions[k], cmap="gray")
        ax.set_title(f"Rank {k}\n{metrics[idx]['psnr']:.1f} dB")
        ax.axis("off")
        
    # Panel 7: Singular value spectrum (decay)
    ax_spec = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    ax_spec.semilogy(range(1, len(S) + 1), S, color="#1f77b4", lw=2)
    ax_spec.grid(True, linestyle="--", alpha=0.6)
    ax_spec.set_title("Singular Value Spectrum (Log Scale)")
    ax_spec.set_xlabel("Singular Value Index")
    ax_spec.set_ylabel("Singular Value Magnitude")
    
    # Panel 8: Cumulative energy
    ax_energy = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    ax_energy.plot(range(1, len(S) + 1), cum_energy, color="#2ca02c", lw=2)
    ax_energy.axhline(90, color="red", linestyle=":", label="90% Threshold")
    ax_energy.axhline(99, color="orange", linestyle=":", label="99% Threshold")
    ax_energy.grid(True, linestyle="--", alpha=0.6)
    ax_energy.set_title("Cumulative Spectral Energy (%)")
    ax_energy.set_xlabel("Rank (Components)")
    ax_energy.set_ylabel("Variance Explained (%)")
    ax_energy.legend(loc="lower right")
    
    # Panel 9: Compression vs Quality (PSNR vs Compression Ratio)
    ax_tradeoff = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    ranks = [m["rank"] for m in metrics]
    psnrs = [m["psnr"] for m in metrics]
    comp_ratios = [m["comp_ratio"] for m in metrics]
    ax_tradeoff.plot(ranks, psnrs, "o-", color="#d62728", lw=2, label="PSNR (dB)")
    ax_tradeoff.set_xlabel("Rank $k$")
    ax_tradeoff.set_ylabel("PSNR (dB)", color="#d62728")
    ax_tradeoff.tick_params(axis="y", labelcolor="#d62728")
    ax_tradeoff.grid(True, linestyle="--", alpha=0.6)
    
    ax_comp = ax_tradeoff.twinx()
    ax_comp.plot(ranks, comp_ratios, "s--", color="#9467bd", lw=2, label="Comp. Ratio")
    ax_comp.set_ylabel("Compression Factor (x)", color="#9467bd")
    ax_comp.tick_params(axis="y", labelcolor="#9467bd")
    ax_tradeoff.set_title("Rank vs Quality & Compression")
    
    plt.tight_layout()
    fig_path = os.path.join("outputs", "figures", "svd_compression_analysis.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved SVD analysis figure to {fig_path}")

if __name__ == "__main__":
    run_svd_analysis()
