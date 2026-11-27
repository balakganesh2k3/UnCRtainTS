#!/usr/bin/env python3
"""
Step 0 premise check for the full-covariance idea.

Question: are the errors of the 13 Sentinel-2 bands correlated, once each error is divided by the
standard deviation the (diagonal) model predicted for it?
If yes, a diagonal covariance misses something and the full-covariance idea is worth trying.

Input : folders with exports written by test_reconstruct.py, i.e. files
            img-<n>_pred.npy    shape (13, H, W)
            img-<n>_target.npy  shape (13, H, W)
            img-<n>_covar.npy   shape (13, 13, H, W)   (diagonal model: only the diagonal is non-zero)
        One folder per group you want to compare (for example one folder per cloud-coverage bin,
        produced by running the test script with different --min_cov / --max_cov).

Usage : python step0_residual_corr.py --bin all=path/to/export/epoch_1/test --out step0_out
        python step0_residual_corr.py --bin clear=dirA --bin partial=dirB --bin heavy=dirC --out step0_out

Outputs (in --out): heatmaps.png, summary.csv, corr_<bin>.csv, zstd_<bin>.csv, report.txt

Notes
 * standardised residual z_k = (target_k - pred_k) / sqrt(var_k), var_k = diagonal of the exported covariance.
   If pred, target and covar are all in the same units (all rescaled, or all not), z is unit-free.
   A z standard deviation far from 1 means the variances are over- or under-confident, OR the units differ.
 * "pooled" = all pixels of all images together. "per-image" = correlation computed inside each image,
   then averaged. If the two disagree a lot, the pooled value is driven by differences between images
   (e.g. overall brightness), not by errors within an image.
 * Uses numpy only for the maths (matplotlib only for the figure). np.load is called with allow_pickle=False.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
K = len(BANDS)
I_B4, I_B8 = BANDS.index('B4'), BANDS.index('B8')  # the NDVI pair


def corr_from_cov(c):
    d = np.sqrt(np.diag(c))
    d[d == 0] = np.nan
    return c / np.outer(d, d)


def load_one(pred_path, min_var):
    """Return standardised residuals as array (13, n_valid_pixels) for one image."""
    target_path = pred_path.replace('_pred.npy', '_target.npy')
    covar_path = pred_path.replace('_pred.npy', '_covar.npy')
    for p in (target_path, covar_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f'missing {p}')
    pred = np.load(pred_path, allow_pickle=False)
    target = np.load(target_path, allow_pickle=False)
    covar = np.load(covar_path, allow_pickle=False)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[0] != K:
        raise ValueError(f'unexpected pred/target shapes {pred.shape} / {target.shape} in {pred_path} '
                         f'(expected ({K}, H, W))')
    h, w = pred.shape[1:]
    covar = covar.reshape(K, K, h, w)  # tolerate leading singleton dimensions
    var = np.stack([covar[k, k] for k in range(K)], 0)
    z = (target.astype(np.float64) - pred) / np.sqrt(np.maximum(var, min_var))
    z = z.reshape(K, -1)
    keep = np.isfinite(z).all(0) & (var.reshape(K, -1) > min_var).all(0)
    return z[:, keep], int(keep.size - keep.sum())


def analyse(folder, min_var, max_images):
    files = sorted(glob.glob(os.path.join(folder, 'img-*_pred.npy')),
                   key=lambda s: int(os.path.basename(s).split('-')[1].split('_')[0]))
    if not files:
        raise SystemExit(f'no img-*_pred.npy files found in {folder}')
    if max_images:
        files = files[:max_images]
    n = 0
    s1 = np.zeros(K)
    s2 = np.zeros((K, K))
    per_image = []
    dropped = 0
    for f in files:
        z, d = load_one(f, min_var)
        dropped += d
        if z.shape[1] < 100:
            continue
        n += z.shape[1]
        s1 += z.sum(1)
        s2 += z @ z.T
        per_image.append(np.corrcoef(z))
    mean = s1 / n
    cov = s2 / n - np.outer(mean, mean)
    pooled = corr_from_cov(cov)
    zstd = np.sqrt(np.diag(cov))
    per_img = np.nanmean(np.stack(per_image), 0) if per_image else np.full((K, K), np.nan)
    return dict(n_images=len(per_image), n_pixels=n, dropped=dropped, pooled=pooled, per_image=per_img, zstd=zstd)


def summarise(c):
    off = c[~np.eye(K, dtype=bool)]
    return dict(mean_offdiag=float(np.nanmean(off)), median_offdiag=float(np.nanmedian(off)),
                share_abs_gt_0_3=float(np.nanmean(np.abs(off) > 0.3)),
                share_abs_gt_0_5=float(np.nanmean(np.abs(off) > 0.5)),
                corr_B4_B8=float(c[I_B4, I_B8]))


def write_matrix(path, m, header=True):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([''] + BANDS)
        for name, row in zip(BANDS, m):
            w.writerow([name] + [f'{v:.4f}' for v in row])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', action='append', required=True, metavar='NAME=FOLDER',
                    help='group name and export folder; repeat for several groups (e.g. cloud-coverage bins)')
    ap.add_argument('--out', default='step0_out')
    ap.add_argument('--min-var', type=float, default=1e-12, help='pixels with a smaller predicted variance are dropped')
    ap.add_argument('--max-images', type=int, default=0, help='use only the first N images per group (0 = all)')
    args = ap.parse_args()

    groups = []
    for item in args.bin:
        if '=' not in item:
            sys.exit(f'--bin needs NAME=FOLDER, got {item!r}')
        name, folder = item.split('=', 1)
        groups.append((name, folder))
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for name, folder in groups:
        print(f'[{name}] reading {folder} ...')
        results[name] = analyse(folder, args.min_var, args.max_images)

    # tables
    rows = []
    lines = []
    for name, r in results.items():
        sp, si = summarise(r['pooled']), summarise(r['per_image'])
        write_matrix(os.path.join(args.out, f'corr_{name}.csv'), r['pooled'])
        with open(os.path.join(args.out, f'zstd_{name}.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(BANDS)
            w.writerow([f'{v:.4f}' for v in r['zstd']])
        rows.append([name, r['n_images'], r['n_pixels'], f"{sp['mean_offdiag']:.3f}", f"{sp['median_offdiag']:.3f}",
                     f"{sp['share_abs_gt_0_5']:.2f}", f"{sp['corr_B4_B8']:.3f}",
                     f"{si['mean_offdiag']:.3f}", f"{si['corr_B4_B8']:.3f}",
                     f"{np.min(r['zstd']):.2f}-{np.max(r['zstd']):.2f}"])
        lines.append(f"{name}: {r['n_images']} images, {r['n_pixels']:,} pixels ({r['dropped']:,} dropped as invalid)\n"
                     f"   pooled   : mean off-diagonal corr {sp['mean_offdiag']:.2f}, median {sp['median_offdiag']:.2f}, "
                     f"share |corr|>0.5: {100 * sp['share_abs_gt_0_5']:.0f}%, B4-B8 {sp['corr_B4_B8']:.2f}\n"
                     f"   per-image: mean off-diagonal corr {si['mean_offdiag']:.2f}, B4-B8 {si['corr_B4_B8']:.2f}\n"
                     f"   z std per band (1.0 = variances are right): {np.round(r['zstd'], 2).tolist()}")
    head = ['group', 'images', 'pixels', 'mean_offdiag(pooled)', 'median_offdiag(pooled)', 'share|corr|>0.5(pooled)',
            'B4-B8(pooled)', 'mean_offdiag(per-image)', 'B4-B8(per-image)', 'z_std_range']
    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)

    caution = []
    for name, r in results.items():
        if r['n_images'] < 30:
            caution.append(f'{name}: only {r["n_images"]} images, treat as a rough indication.')
        if r['zstd'].min() < 0.5 or r['zstd'].max() > 2:
            caution.append(f'{name}: z std per band is far from 1 ({r["zstd"].min():.2f} to {r["zstd"].max():.2f}). '
                           'Either the predicted variances are over/under-confident or pred/target/covar are in different '
                           'units (check how scale_by is undone before trusting calibration numbers).')
        sp, si = summarise(r['pooled']), summarise(r['per_image'])
        if abs(sp['mean_offdiag'] - si['mean_offdiag']) > 0.2:
            caution.append(f'{name}: pooled and per-image correlations differ by more than 0.2, so part of the pooled '
                           'correlation comes from differences between images and not from errors inside an image.')
    with open(os.path.join(args.out, 'report.txt'), 'w') as fh:
        fh.write('Step 0: correlation of standardised residuals between the 13 bands\n\n' + '\n\n'.join(lines))
        if caution:
            fh.write('\n\nCautions:\n' + '\n'.join(' - ' + c for c in caution))
        fh.write('\n')
    print(open(os.path.join(args.out, 'report.txt')).read())

    # figure: one heatmap per group, shared colour scale
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n + 0.8, 5.2), squeeze=False)
    im = None
    for ax, (name, r) in zip(axes[0], results.items()):
        im = ax.imshow(r['pooled'], vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(BANDS, rotation=90, fontsize=8); ax.set_yticklabels(BANDS, fontsize=8)
        sp = summarise(r['pooled'])
        ax.set_title(f"{name}\n{r['n_images']} images, mean off-diag {sp['mean_offdiag']:.2f}", fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label='correlation of standardised residuals')
    fig.savefig(os.path.join(args.out, 'heatmaps.png'), dpi=160, bbox_inches='tight')
    print(f'saved {os.path.join(args.out, "heatmaps.png")}, summary.csv, report.txt and per-group CSVs in {args.out}/')


if __name__ == '__main__':
    main()
