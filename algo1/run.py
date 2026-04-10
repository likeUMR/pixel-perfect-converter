#!/usr/bin/env python3
"""
Algorithm 1: 撒豆算法 (Bean Seed)
===================================
核心思路：
  密集撒入检测点 → 每点寻找最大纯色正方块 → 合并相似方块
  → 推断像素尺寸（去离群值取中位数）→ 归一化 → 分裂过大块
  → 网格对齐 → 填充空格 → 输出

每步保存一张中间图像。

用法：
  python run.py <输入图片> [--pixel-size N] [--stride N] [--var-thresh N]
"""

import sys
import argparse
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter
from collections import Counter

# ── 引入公共工具 ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import (remove_white_background, get_content_bbox,
                           detect_pixel_size, save_step,
                           make_base_canvas, draw_grid, _get_font)

# ──────────────────────────────────────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────────────────────────────────────
DEF_STRIDE     = 3     # 种子点间距（像素）
DEF_VAR_THRESH = 450   # 3通道方差之和上限（判定"纯色"）
DEF_MAX_SQ     = 38    # 最大搜索正方块边长


# ──────────────────────────────────────────────────────────────────────────────
# 核心工具函数
# ──────────────────────────────────────────────────────────────────────────────

def build_variance_maps(rgb: np.ndarray, valid: np.ndarray,
                         sizes: list[int]) -> dict[int, np.ndarray]:
    """
    对每种窗口尺寸用 uniform_filter 计算每像素处的局部颜色方差图。
    valid 区域之外设为极大值，排除透明背景的干扰。
    """
    rgb_f = rgb.astype(np.float32)
    valid_f = valid.astype(np.float32)
    maps = {}
    for S in sizes:
        var = np.zeros(rgb.shape[:2], dtype=np.float32)
        for c in range(3):
            ch = rgb_f[:, :, c]
            m1 = uniform_filter(ch, S, mode="nearest")
            m2 = uniform_filter(ch ** 2, S, mode="nearest")
            var += np.maximum(0.0, m2 - m1 ** 2)
        var[~valid] = 1e9
        maps[S] = var
    return maps


def find_best_sizes(seeds_y: np.ndarray, seeds_x: np.ndarray,
                     var_maps: dict[int, np.ndarray],
                     sizes_desc: list[int],
                     threshold: float) -> np.ndarray:
    """
    对每个种子点，从最大尺寸往下找，返回该点首个满足方差<阈值的正方块尺寸。
    找不到则为 0。
    """
    n = len(seeds_y)
    best = np.zeros(n, dtype=np.int32)
    found = np.zeros(n, dtype=bool)
    for S in sizes_desc:
        if found.all():
            break
        vm = var_maps[S]
        seed_var = vm[seeds_y, seeds_x]
        newly = (~found) & (seed_var < threshold)
        best[newly] = S
        found[newly] = True
    return best


def merge_squares(squares: list[dict], ps: int) -> list[dict]:
    """
    按方差升序排列，贪心去重：若两方块中心距离 < pixel_size*0.7 则视为重复，保留方差小的。
    """
    if not squares:
        return []
    squares = sorted(squares, key=lambda s: s["var"])
    merged = []
    for sq in squares:
        x0, y0, x1, y1 = sq["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        dup = False
        for m in merged:
            mx0, my0, mx1, my1 = m["bbox"]
            mcx, mcy = (mx0 + mx1) / 2, (my0 + my1) / 2
            if max(abs(cx - mcx), abs(cy - mcy)) < ps * 0.65:
                dup = True
                break
        if not dup:
            merged.append(sq)
    return merged


def infer_pixel_size(squares: list[dict]) -> tuple[int, np.ndarray]:
    """
    从方块尺寸分布中去除 IQR 离群值，取中位数作为 pixel_size。
    返回 (pixel_size, inlier_mask)。
    """
    sizes = np.array([s["size"] for s in squares])
    q1, q3 = np.percentile(sizes, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inlier_mask = (sizes >= max(2, lo)) & (sizes <= hi)
    ps = int(round(np.median(sizes[inlier_mask]))) if inlier_mask.any() else int(np.median(sizes))
    return ps, inlier_mask


def find_grid_offset(squares: list[dict], pixel_size: int) -> tuple[int, int]:
    """
    通过所有方块中心的 mod pixel_size 分布，用直方图众数找最佳全局偏移。
    """
    if not squares:
        return 0, 0
    half = pixel_size / 2
    cx_arr = np.array([(s["bbox"][0] + s["bbox"][2]) / 2 - half for s in squares])
    cy_arr = np.array([(s["bbox"][1] + s["bbox"][3]) / 2 - half for s in squares])
    mod_x = cx_arr % pixel_size
    mod_y = cy_arr % pixel_size
    hx, _ = np.histogram(mod_x, bins=pixel_size, range=(0, pixel_size))
    hy, _ = np.histogram(mod_y, bins=pixel_size, range=(0, pixel_size))
    return int(np.argmax(hx)), int(np.argmax(hy))


# ──────────────────────────────────────────────────────────────────────────────
# 可视化辅助
# ──────────────────────────────────────────────────────────────────────────────

def draw_squares_on_dark(squares: list[dict], w: int, h: int,
                          border=(200, 200, 200, 80)) -> Image.Image:
    vis = Image.new("RGBA", (w, h), (38, 38, 42, 255))
    d = ImageDraw.Draw(vis)
    for sq in squares:
        x0, y0, x1, y1 = sq["bbox"]
        r, g, b = sq["color"]
        d.rectangle([x0, y0, x1 - 1, y1 - 1],
                    fill=(r, g, b, 210), outline=border, width=1)
    return vis


def size_histogram_img(sizes_all, sizes_inlier, pixel_size: int,
                        lo: float, hi: float) -> Image.Image:
    """绘制简易尺寸直方图（PIL 实现）。"""
    W, H = 640, 280
    img = Image.new("RGBA", (W, H + 50), (28, 28, 34, 255))
    d = ImageDraw.Draw(img)
    counter = Counter(sizes_all.tolist())
    if not counter:
        return img
    sz_min, sz_max = min(counter), max(counter)
    sz_range = max(1, sz_max - sz_min)
    max_cnt = max(counter.values())
    font = _get_font(14)
    for sz, cnt in sorted(counter.items()):
        bx = int((sz - sz_min) / sz_range * (W - 60)) + 30
        bh = int(cnt / max_cnt * (H - 20))
        is_out = sz < lo or sz > hi
        clr = (80, 80, 180, 255) if is_out else (50, 190, 100, 255)
        if sz == pixel_size:
            clr = (255, 200, 50, 255)
        d.rectangle([bx - 3, H - bh, bx + 3, H], fill=clr)
    d.text((10, H + 8),
           f"Pixel size: {pixel_size}px  |  Yellow=selected  Blue=outlier  "
           f"Inlier range=[{lo:.0f},{hi:.0f}]",
           fill=(210, 210, 190, 255), font=font)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    src = Path(args.input)
    OUT = Path(__file__).parent / "steps"
    OUT.mkdir(exist_ok=True)
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  Algorithm 1: Bean Seed (撒豆算法)")
    print(f"  Input: {src}")
    print(f"{sep}")

    # ── Step 1: 去白底 ────────────────────────────────────────────────────────
    print("\n[1/10] Remove white background...")
    img = Image.open(src).convert("RGB")
    img_nobg = remove_white_background(img, threshold=235)
    save_step(img_nobg, OUT, "01_nobg", "Step1: White background removed (transparent)")

    data = np.array(img_nobg)
    valid = data[:, :, 3] > 0
    rgb = data[:, :, :3]
    h, w = rgb.shape[:2]
    bx0, by0, bx1, by1 = get_content_bbox(valid, padding=10)
    print(f"  Content bbox: ({bx0},{by0})→({bx1},{by1})  size={bx1-bx0}×{by1-by0}")

    # ── Step 2: 检测 pixel_size（或使用传入值）────────────────────────────────
    if args.pixel_size:
        pixel_size = args.pixel_size
        print(f"\n[2/10] Using specified pixel_size={pixel_size}")
    else:
        print("\n[2/10] Detecting pixel size...", end=" ", flush=True)
        pixel_size, _, _ = detect_pixel_size(img_nobg, verbose=True)
        print(f"-> {pixel_size}px")

    stride = args.stride
    var_thresh = args.var_thresh
    max_sq = min(args.max_sq, pixel_size * 3)

    # ── Step 3: 撒豆（种子点）────────────────────────────────────────────────
    print(f"\n[3/10] Seeding (stride={stride})...")
    sy_arr = np.arange(by0, by1, stride)
    sx_arr = np.arange(bx0, bx1, stride)
    sy_g, sx_g = np.meshgrid(sy_arr, sx_arr, indexing="ij")
    sy_flat, sx_flat = sy_g.ravel(), sx_g.ravel()
    on_valid = valid[sy_flat, sx_flat]
    seeds_y = sy_flat[on_valid]
    seeds_x = sx_flat[on_valid]
    print(f"  Seeds: {len(seeds_y)}")

    seed_vis = make_base_canvas(img_nobg)
    sd = ImageDraw.Draw(seed_vis)
    for py, px in zip(seeds_y[:3000].tolist(), seeds_x[:3000].tolist()):
        sd.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(255, 60, 60, 180))
    save_step(seed_vis, OUT, "02_seeds",
              f"Step2: {len(seeds_y)} seed points  stride={stride}px")

    # ── Step 4: 每个种子点找最大纯色正方块 ───────────────────────────────────
    print(f"\n[4/10] Finding largest pure square per seed (max={max_sq}px, var<{var_thresh})...")
    sizes_desc = list(range(max_sq, 2, -1))
    print("  Building variance maps...", end=" ", flush=True)
    var_maps = build_variance_maps(rgb, valid, sizes_desc)
    print("done")
    print("  Searching sizes...", end=" ", flush=True)
    best_sizes = find_best_sizes(seeds_y, seeds_x, var_maps, sizes_desc, var_thresh)
    print("done")

    raw_squares = []
    for py, px, sz in zip(seeds_y.tolist(), seeds_x.tolist(), best_sizes.tolist()):
        if sz < 3:
            continue
        x0 = max(0, px - sz // 2)
        y0 = max(0, py - sz // 2)
        x1, y1 = x0 + sz, y0 + sz
        if x1 > w or y1 > h:
            continue
        cell = rgb[y0:y1, x0:x1]
        cell_v = valid[y0:y1, x0:x1]
        pixels = cell[cell_v]
        if len(pixels) < sz * sz * 0.25:
            continue
        color = tuple(np.median(pixels, axis=0).astype(np.uint8))
        raw_squares.append({
            "bbox": (x0, y0, x1, y1),
            "size": sz,
            "color": color,
            "var": float(var_maps[sz][py, px]),
        })
    print(f"  Raw squares found: {len(raw_squares)}")
    save_step(draw_squares_on_dark(raw_squares, w, h), OUT, "03_raw_squares",
              f"Step3: {len(raw_squares)} raw pure squares (each seed→largest pure box)")

    # ── Step 5: 合并重叠方块 ─────────────────────────────────────────────────
    print(f"\n[5/10] Merging overlapping squares...")
    merged = merge_squares(raw_squares, pixel_size)
    print(f"  {len(raw_squares)} → {len(merged)} squares after merge")
    save_step(draw_squares_on_dark(merged, w, h, border=(60, 220, 60, 150)),
              OUT, "04_merged",
              f"Step4: {len(merged)} unique squares after deduplication")

    # ── Step 6: 推断像素尺寸 ─────────────────────────────────────────────────
    print(f"\n[6/10] Inferring pixel size from merged squares...")
    sizes_all = np.array([s["size"] for s in merged])
    q1, q3 = np.percentile(sizes_all, [25, 75])
    iqr = q3 - q1
    lo, hi_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inlier_mask = (sizes_all >= max(2, lo)) & (sizes_all <= hi_bound)
    inlier_sizes = sizes_all[inlier_mask]
    inferred_ps = int(round(np.median(inlier_sizes))) if inlier_mask.any() else pixel_size
    print(f"  Distribution: min={sizes_all.min()} max={sizes_all.max()} "
          f"median={np.median(sizes_all):.1f}")
    print(f"  Inlier range: [{lo:.1f}, {hi_bound:.1f}]  → pixel_size={inferred_ps}px")
    # 用推断值覆盖（如果合理）
    if abs(inferred_ps - pixel_size) <= 2:
        pixel_size = inferred_ps
    print(f"  Final pixel_size: {pixel_size}px")

    hist_img = size_histogram_img(sizes_all, inlier_sizes, pixel_size, lo, hi_bound)
    save_step(hist_img, OUT, "05_size_histogram",
              f"Step5: Size distribution → inferred pixel_size={pixel_size}px")

    # ── Step 7: 归一化为 pixel_size 方块 ─────────────────────────────────────
    print(f"\n[7/10] Normalizing inlier squares to {pixel_size}×{pixel_size}...")
    norm_squares = []
    for sq, is_in in zip(merged, inlier_mask):
        if not is_in:
            continue
        x0, y0, x1, y1 = sq["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        nx0 = int(round(cx - pixel_size / 2))
        ny0 = int(round(cy - pixel_size / 2))
        nx1, ny1 = nx0 + pixel_size, ny0 + pixel_size
        if nx0 < 0 or ny0 < 0 or nx1 > w or ny1 > h:
            continue
        pixels = rgb[ny0:ny1, nx0:nx1][valid[ny0:ny1, nx0:nx1]]
        if len(pixels) < pixel_size * pixel_size * 0.25:
            continue
        norm_squares.append({
            "bbox": (nx0, ny0, nx1, ny1),
            "size": pixel_size,
            "color": tuple(np.median(pixels, axis=0).astype(np.uint8)),
        })
    save_step(draw_squares_on_dark(norm_squares, w, h, border=(150, 150, 255, 80)),
              OUT, "06_normalized",
              f"Step6: {len(norm_squares)} squares normalized to {pixel_size}px (may overlap)")

    # ── Step 8: 处理过大离群块（分裂）────────────────────────────────────────
    print(f"\n[8/10] Splitting oversized outlier squares...")
    split_squares = list(norm_squares)
    outliers_big = [sq for sq, is_in in zip(merged, inlier_mask)
                    if not is_in and sq["size"] > hi_bound]
    split_count = 0
    for sq in outliers_big:
        sz = sq["size"]
        k = max(2, int(round(sz / pixel_size)))
        if k > 6:
            continue  # 太大，跳过
        x0, y0, x1, y1 = sq["bbox"]
        sub = pixel_size
        for ky in range(k):
            for kx in range(k):
                sx0, sy0 = x0 + kx * sub, y0 + ky * sub
                sx1, sy1 = sx0 + sub, sy0 + sub
                if sx1 > w or sy1 > h:
                    continue
                pixels = rgb[sy0:sy1, sx0:sx1][valid[sy0:sy1, sx0:sx1]]
                if len(pixels) < sub * sub * 0.2:
                    continue
                split_squares.append({
                    "bbox": (sx0, sy0, sx1, sy1),
                    "size": pixel_size,
                    "color": tuple(np.median(pixels, axis=0).astype(np.uint8)),
                })
                split_count += 1
    print(f"  Split {len(outliers_big)} oversized outliers into {split_count} sub-pixels")
    save_step(draw_squares_on_dark(split_squares, w, h, border=(200, 180, 60, 80)),
              OUT, "07_split",
              f"Step7: {len(split_squares)} pixels after splitting oversized")

    # ── Step 9: 网格对齐 ──────────────────────────────────────────────────────
    print(f"\n[9/10] Grid alignment...")
    ox, oy = find_grid_offset(split_squares, pixel_size)
    print(f"  Best offset: ({ox}, {oy})")

    # 每个方块中心 → 最近格子
    grid: dict[tuple, list] = {}
    for sq in split_squares:
        x0, y0, x1, y1 = sq["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        gx = int(round((cx - ox - pixel_size / 2) / pixel_size))
        gy = int(round((cy - oy - pixel_size / 2) / pixel_size))
        grid.setdefault((gx, gy), []).append(sq)

    aligned: dict[tuple, tuple] = {}  # (gx,gy) → color (R,G,B)
    conflicts = 0
    for (gx, gy), sqs in grid.items():
        ax0 = ox + gx * pixel_size
        ay0 = oy + gy * pixel_size
        if ax0 < 0 or ay0 < 0 or ax0 + pixel_size > w or ay0 + pixel_size > h:
            continue
        if len(sqs) > 1:
            conflicts += 1
            color = tuple(np.median([list(s["color"]) for s in sqs], axis=0).astype(np.uint8))
        else:
            color = sqs[0]["color"]
        aligned[(gx, gy)] = color
    print(f"  Grid cells: {len(aligned)}, conflicts merged: {conflicts}")

    # 可视化对齐结果
    align_vis = Image.new("RGBA", (w, h), (38, 38, 42, 255))
    avd = ImageDraw.Draw(align_vis)
    for (gx, gy), color in aligned.items():
        ax0 = ox + gx * pixel_size
        ay0 = oy + gy * pixel_size
        ax1, ay1 = ax0 + pixel_size, ay0 + pixel_size
        r, g2, b = color
        avd.rectangle([ax0, ay0, ax1 - 1, ay1 - 1], fill=(r, g2, b, 255))
    draw_grid(avd, w, h, ox, oy, pixel_size, color=(255, 255, 255, 45))
    save_step(align_vis, OUT, "08_grid_aligned",
              f"Step8: Grid-aligned  offset=({ox},{oy})  {pixel_size}px  "
              f"{len(aligned)} cells  {conflicts} conflicts")

    # ── Step 10: 填充空格 ─────────────────────────────────────────────────────
    print(f"\n[10/10] Filling empty neighbor cells...")
    occupied = set(aligned.keys())
    # 找到有内容邻格的空格（只在角色范围内）
    to_fill = set()
    for (gx, gy) in occupied:
        for dgy, dgx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            key = (gx + dgx, gy + dgy)
            if key in occupied:
                continue
            ax0 = ox + key[0] * pixel_size
            ay0 = oy + key[1] * pixel_size
            ax1, ay1 = ax0 + pixel_size, ay0 + pixel_size
            if ax0 < bx0 or ay0 < by0 or ax1 > bx1 or ay1 > by1:
                continue
            if ax0 < 0 or ay0 < 0 or ax1 > w or ay1 > h:
                continue
            cell_v = valid[ay0:ay1, ax0:ax1]
            if cell_v.any():
                to_fill.add(key)

    filled = dict(aligned)
    fill_count = 0
    for key in to_fill:
        ax0 = ox + key[0] * pixel_size
        ay0 = oy + key[1] * pixel_size
        # 优先用图像实际颜色（取非透明像素中值）
        cell_rgb = rgb[ay0:ay0 + pixel_size, ax0:ax0 + pixel_size]
        cell_v = valid[ay0:ay0 + pixel_size, ax0:ax0 + pixel_size]
        pixels = cell_rgb[cell_v]
        if len(pixels) >= 4:
            filled[key] = tuple(np.median(pixels, axis=0).astype(np.uint8))
        else:
            # 用随机邻格颜色填充
            nbrs = [filled[k] for k in [
                (key[0] - 1, key[1]), (key[0] + 1, key[1]),
                (key[0], key[1] - 1), (key[0], key[1] + 1)
            ] if k in filled]
            if nbrs:
                filled[key] = random.choice(nbrs)
        fill_count += 1
    print(f"  Filled {fill_count} empty cells")

    fill_vis = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    fvd = ImageDraw.Draw(fill_vis)
    for (gx, gy), color in filled.items():
        ax0 = ox + gx * pixel_size
        ay0 = oy + gy * pixel_size
        r, g2, b = color
        fvd.rectangle([ax0, ay0, ax0 + pixel_size - 1, ay0 + pixel_size - 1],
                      fill=(r, g2, b, 255))
    save_step(fill_vis, OUT, "09_filled",
              f"Step9: {fill_count} empty cells filled (image color or random neighbor)")

    # ── Final: 合成最终输出 ───────────────────────────────────────────────────
    final = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    fnd = ImageDraw.Draw(final)
    for (gx, gy), color in filled.items():
        ax0 = ox + gx * pixel_size
        ay0 = oy + gy * pixel_size
        r, g2, b = color
        fnd.rectangle([ax0, ay0, ax0 + pixel_size - 1, ay0 + pixel_size - 1],
                      fill=(r, g2, b, 255))
    save_step(final, OUT, "10_final",
              f"Algorithm1 FINAL: pixel_size={pixel_size}px  "
              f"offset=({ox},{oy})  cells={len(filled)}")

    print(f"\nDone! All steps saved to: {OUT}")
    print(f"Final pixel_size={pixel_size}  offset=({ox},{oy})  "
          f"total_cells={len(filled)}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Algorithm 1: Bean Seed")
    p.add_argument("input", help="Input image path")
    p.add_argument("--pixel-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=DEF_STRIDE)
    p.add_argument("--var-thresh", type=float, default=DEF_VAR_THRESH)
    p.add_argument("--max-sq", type=int, default=DEF_MAX_SQ)
    main(p.parse_args())
