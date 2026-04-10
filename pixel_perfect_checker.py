#!/usr/bin/env python3
"""
Pixel Perfect Checker
=====================
对 AI 生成的白底像素风格图片进行分析与修复：
  1. 去除白底 -> 透明背景
  2. 检测像素网格尺寸（逻辑像素 = N×N 屏幕像素）及最佳偏移
  3. 逐格检查纯色度
  4. 生成 pixel-perfect 修复版本

用法：
  python pixel_perfect_checker.py <输入图片> [选项]

示例：
  python pixel_perfect_checker.py D:/Project/tools/image.png
  python pixel_perfect_checker.py image.png --max-pixel-size 16 --purity-threshold 25
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────────────
# 1. 去除白底
# ──────────────────────────────────────────────────────────────────────────────

def remove_white_background(img: Image.Image, white_threshold: int = 235) -> Image.Image:
    """
    使用 BFS 泛洪填充从图像四边出发，标记背景白色区域并设为透明。
    只删除背景中的白色，保留角色内部的白色细节。
    """
    img_rgba = img.convert("RGBA")
    data = np.array(img_rgba, dtype=np.uint8)
    h, w = data.shape[:2]

    r = data[:, :, 0].astype(np.int32)
    g = data[:, :, 1].astype(np.int32)
    b = data[:, :, 2].astype(np.int32)

    # 判断"白色"像素：三通道均超过阈值
    is_white = (r >= white_threshold) & (g >= white_threshold) & (b >= white_threshold)

    # BFS 从四边出发标记背景
    background = np.zeros((h, w), dtype=bool)
    from collections import deque
    queue = deque()

    def enqueue(y, x):
        if 0 <= y < h and 0 <= x < w and not background[y, x] and is_white[y, x]:
            background[y, x] = True
            queue.append((y, x))

    for x in range(w):
        enqueue(0, x)
        enqueue(h - 1, x)
    for y in range(h):
        enqueue(y, 0)
        enqueue(y, w - 1)

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in dirs:
            enqueue(cy + dy, cx + dx)

    result = data.copy()
    result[background, 3] = 0
    return Image.fromarray(result, "RGBA")


# ──────────────────────────────────────────────────────────────────────────────
# 2. 检测像素网格（跳变间距和声分析 + 梯度偏移精修）
# ──────────────────────────────────────────────────────────────────────────────

def _get_content_bbox(valid: np.ndarray, padding: int = 4) -> tuple[int, int, int, int]:
    """返回非透明像素的边界框 (x0, y0, x1, y1)，带 padding。"""
    ys, xs = np.where(valid)
    if len(ys) == 0:
        h, w = valid.shape
        return 0, 0, w, h
    h, w = valid.shape
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(w, int(xs.max()) + padding + 1)
    y1 = min(h, int(ys.max()) + padding + 1)
    return x0, y0, x1, y1


def _transition_harmony_score(rgb: np.ndarray, valid: np.ndarray,
                               diff_threshold: float = 35.0,
                               max_size: int = 32) -> dict[int, int]:
    """
    统计颜色跳变的间距分布，对每个候选尺寸 N 计算"和声分数"：
    N 的倍数（N, 2N, 3N...）出现次数之和。越高越好。
    """
    h, w = rgb.shape[:2]
    gap_counter: Counter = Counter()

    # 水平扫描
    for y in range(0, h, max(1, h // 120)):
        row = rgb[y].astype(float)
        vrow = valid[y]
        prev = None
        for x in range(1, w):
            if not (vrow[x] and vrow[x - 1]):
                continue
            d = float(np.sum(np.abs(row[x] - row[x - 1])))
            if d > diff_threshold:
                if prev is not None:
                    gap = x - prev
                    if 2 <= gap <= max_size * 3:
                        gap_counter[gap] += 1
                prev = x

    # 垂直扫描
    for x in range(0, w, max(1, w // 120)):
        col = rgb[:, x].astype(float)
        vcol = valid[:, x]
        prev = None
        for y in range(1, h):
            if not (vcol[y] and vcol[y - 1]):
                continue
            d = float(np.sum(np.abs(col[y] - col[y - 1])))
            if d > diff_threshold:
                if prev is not None:
                    gap = y - prev
                    if 2 <= gap <= max_size * 3:
                        gap_counter[gap] += 1
                prev = y

    # 对每个候选尺寸计算和声分数
    scores: dict[int, int] = {}
    for sz in range(2, max_size + 1):
        scores[sz] = sum(gap_counter[sz * k] for k in range(1, max_size * 3 // sz + 1))
    return scores


def _gradient_score_fast(rgb: np.ndarray, valid: np.ndarray,
                          size: int, oy: int, ox: int) -> float:
    """
    "格内平均梯度 / 边界平均梯度"比值——越小说明对齐越好。
    用 numpy 向量化，避免 Python 循环。
    """
    rgb_f = rgb.astype(np.float32)
    h, w = rgb.shape[:2]

    # 水平差分
    h_diff = np.sum(np.abs(rgb_f[:, 1:] - rgb_f[:, :-1]), axis=2)
    h_valid = valid[:, 1:] & valid[:, :-1]
    xs = np.arange(w - 1)
    h_bnd = ((xs - ox + 1) % size == 0)[np.newaxis, :]
    h_int = h_diff[h_valid & ~h_bnd]
    h_bnd_vals = h_diff[h_valid & h_bnd]

    # 垂直差分
    v_diff = np.sum(np.abs(rgb_f[1:] - rgb_f[:-1]), axis=2)
    v_valid = valid[1:] & valid[:-1]
    ys = np.arange(h - 1)
    v_bnd = ((ys - oy + 1) % size == 0)[:, np.newaxis]
    v_int = v_diff[v_valid & ~v_bnd]
    v_bnd_vals = v_diff[v_valid & v_bnd]

    interior = np.concatenate([h_int, v_int])
    boundary = np.concatenate([h_bnd_vals, v_bnd_vals])

    if len(interior) == 0 or len(boundary) == 0:
        return float("inf")
    return float(np.mean(interior)) / (float(np.mean(boundary)) + 1e-6)


def detect_pixel_grid(img_rgba: Image.Image,
                      max_pixel_size: int = 32) -> tuple[int, int, int, dict]:
    """
    检测逻辑像素大小及网格偏移量。

    算法：
      1. 裁剪到角色内容区域（非透明 BBox）
      2. 用"跳变间距和声分析"找到最佳像素尺寸
      3. 用"梯度比率法"在全图上精确查找对齐偏移

    返回：(pixel_size, offset_x, offset_y, harmony_scores)
    """
    data = np.array(img_rgba)
    if data.shape[2] == 4:
        valid_full = data[:, :, 3] > 0
        rgb_full = data[:, :, :3]
    else:
        valid_full = np.ones(data.shape[:2], dtype=bool)
        rgb_full = data[:, :, :3]

    # Step A: 裁剪到角色区域，仅在该区域内检测像素尺寸
    bx0, by0, bx1, by1 = _get_content_bbox(valid_full, padding=2)
    rgb_crop = rgb_full[by0:by1, bx0:bx1]
    valid_crop = valid_full[by0:by1, bx0:bx1]

    print("")
    print("  [detect] Content bbox:", bx0, by0, bx1, by1,
          "->", bx1 - bx0, "x", by1 - by0, "px", flush=True)
    print("  [detect] Scanning transitions...", end=" ", flush=True)

    harmony_scores = _transition_harmony_score(
        rgb_crop, valid_crop, diff_threshold=35, max_size=max_pixel_size
    )
    print("done")

    # 选出和声分数最高的尺寸
    best_size = max(harmony_scores, key=harmony_scores.get)

    # Step B: 在全图中用梯度法精确找到 offset
    print(f"  [detect] Refining offset for size={best_size}...", end=" ", flush=True)
    best_offset_score = float("inf")
    best_ox, best_oy = 0, 0

    # 先在裁剪区域相对坐标内扫描偏移
    for oy in range(best_size):
        for ox in range(best_size):
            s = _gradient_score_fast(rgb_crop, valid_crop, best_size, oy, ox)
            if s < best_offset_score:
                best_offset_score = s
                # 转换回全图坐标
                best_ox = (bx0 + ox) % best_size
                best_oy = (by0 + oy) % best_size

    print("done")
    return best_size, best_ox, best_oy, harmony_scores


# ──────────────────────────────────────────────────────────────────────────────
# 3. 纯色度检测
# ──────────────────────────────────────────────────────────────────────────────

def check_purity(img_rgba: Image.Image, pixel_size: int,
                 offset_x: int, offset_y: int,
                 purity_threshold: int = 30) -> tuple[np.ndarray, float, int, int]:
    """
    遍历每个逻辑像素格，判断格内像素是否纯色。
    返回：(purity_map, purity_ratio, impure_count, total_count)
    """
    data = np.array(img_rgba)
    if data.shape[2] == 4:
        alpha = data[:, :, 3]
        rgb = data[:, :, :3].astype(np.int32)
        valid = alpha > 0
    else:
        rgb = data[:, :, :3].astype(np.int32)
        valid = np.ones(data.shape[:2], dtype=bool)

    h, w = rgb.shape[:2]
    grid_rows = (h - offset_y) // pixel_size
    grid_cols = (w - offset_x) // pixel_size

    purity_map = np.zeros((grid_rows, grid_cols), dtype=bool)
    impure = 0
    total = 0

    for gy in range(grid_rows):
        for gx in range(grid_cols):
            cy = offset_y + gy * pixel_size
            cx = offset_x + gx * pixel_size
            cell_rgb = rgb[cy:cy + pixel_size, cx:cx + pixel_size]
            cell_valid = valid[cy:cy + pixel_size, cx:cx + pixel_size]
            pixels = cell_rgb[cell_valid]
            if len(pixels) == 0:
                continue
            total += 1
            # 通道最大差值 <= 阈值 视为纯色
            max_diff = int((pixels.max(axis=0) - pixels.min(axis=0)).max())
            if max_diff <= purity_threshold:
                purity_map[gy, gx] = True
            else:
                impure += 1

    ratio = 1.0 - (impure / total if total > 0 else 0.0)
    return purity_map, ratio, impure, total


# ──────────────────────────────────────────────────────────────────────────────
# 4. 生成 Pixel-Perfect 版本
# ──────────────────────────────────────────────────────────────────────────────

def generate_pixel_perfect(img_rgba: Image.Image,
                            pixel_size: int,
                            offset_x: int,
                            offset_y: int,
                            min_coverage: float = 0.25) -> tuple[Image.Image, Image.Image]:
    """
    生成两种输出：
      - full:  原始尺寸像素完美版（每格用中值色填充）
      - native: 缩小到逻辑像素分辨率的小图（1格=1像素）

    min_coverage: 格内非透明像素占比至少达到此值才视为有效格，
                  避免边缘半透明格产生脏点。
    """
    data = np.array(img_rgba)
    if data.shape[2] == 4:
        alpha = data[:, :, 3]
        rgb = data[:, :, :3]
        valid = alpha > 0
    else:
        alpha = np.full(data.shape[:2], 255, dtype=np.uint8)
        rgb = data[:, :, :3]
        valid = np.ones(data.shape[:2], dtype=bool)

    h, w = rgb.shape[:2]
    result = np.zeros((h, w, 4), dtype=np.uint8)

    cell_size = pixel_size * pixel_size  # 每格总像素数
    min_pixels = max(1, int(cell_size * min_coverage))

    # 记录逻辑像素颜色用于原生分辨率输出
    grid_rows = (h - offset_y) // pixel_size
    grid_cols = (w - offset_x) // pixel_size
    native_data = np.zeros((grid_rows, grid_cols, 4), dtype=np.uint8)

    for gy in range(grid_rows):
        for gx in range(grid_cols):
            cy = offset_y + gy * pixel_size
            cx = offset_x + gx * pixel_size
            cell_rgb = rgb[cy:cy + pixel_size, cx:cx + pixel_size]
            cell_valid = valid[cy:cy + pixel_size, cx:cx + pixel_size]
            pixels = cell_rgb[cell_valid]

            if len(pixels) < min_pixels:
                # 覆盖率不足，视为透明格
                continue

            dominant = np.median(pixels, axis=0).astype(np.uint8)
            result[cy:cy + pixel_size, cx:cx + pixel_size, :3] = dominant
            result[cy:cy + pixel_size, cx:cx + pixel_size, 3] = 255
            native_data[gy, gx, :3] = dominant
            native_data[gy, gx, 3] = 255

    return Image.fromarray(result, "RGBA"), Image.fromarray(native_data, "RGBA")


# ──────────────────────────────────────────────────────────────────────────────
# 5. 诊断可视化图
# ──────────────────────────────────────────────────────────────────────────────

def generate_diagnostic(img_rgba: Image.Image,
                         purity_map: np.ndarray,
                         pixel_size: int,
                         offset_x: int,
                         offset_y: int,
                         purity_ratio: float,
                         min_coverage: float = 0.25) -> Image.Image:
    """
    叠加格子颜色标注：
      绿色 = 纯色格（颜色均匀）
      红色 = 非纯色格（有抗锯齿/混色）
      透明 = 背景空格（忽略）
    """
    data = np.array(img_rgba)
    valid = data[:, :, 3] > 0 if data.shape[2] == 4 else np.ones(data.shape[:2], dtype=bool)
    cell_size = pixel_size * pixel_size
    min_pixels = max(1, int(cell_size * min_coverage))

    base = img_rgba.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    grid_rows, grid_cols = purity_map.shape
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            cy = offset_y + gy * pixel_size
            cx = offset_x + gx * pixel_size
            cell_valid = valid[cy:cy + pixel_size, cx:cx + pixel_size]
            # 跳过背景（空格/低覆盖率格）
            if cell_valid.sum() < min_pixels:
                continue

            x1, y1 = cx, cy
            x2, y2 = cx + pixel_size - 1, cy + pixel_size - 1

            if purity_map[gy, gx]:
                fill = (0, 255, 0, 45)
                outline = (0, 200, 0, 100)
            else:
                fill = (255, 60, 0, 60)
                outline = (220, 0, 0, 130)

            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline)

    result = Image.alpha_composite(base, overlay)

    # 信息栏
    info_h = 56
    canvas = Image.new("RGBA", (result.width, result.height + info_h), (28, 28, 30, 255))
    canvas.paste(result, (0, 0))
    d = ImageDraw.Draw(canvas)

    text = (f"Grid: {pixel_size}x{pixel_size}px | Offset: ({offset_x},{offset_y}) | "
            f"Purity: {purity_ratio * 100:.1f}% | Green=pure  Red=impure")
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    d.text((12, result.height + 12), text, fill=(255, 240, 180, 255), font=font)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────────────────────────────────────

def process_image(input_path: str,
                  white_threshold: int = 235,
                  max_pixel_size: int = 32,
                  purity_threshold: int = 30,
                  output_dir: str | None = None,
                  save_steps: bool = True) -> dict:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Pixel Perfect Checker")
    print(f"{sep}")
    print(f"  Input : {src}")
    print(f"  Output: {out_dir}")

    # Step 1: 加载
    print(f"\n[1/4] Loading image...", end=" ", flush=True)
    img = Image.open(src).convert("RGB")
    print(f"OK  ({img.width} x {img.height} px)")

    # Step 2: 去除白底
    print(f"[2/4] Removing white background (threshold={white_threshold})...", end=" ", flush=True)
    img_nobg = remove_white_background(img, white_threshold)

    # 统计透明像素数量
    arr = np.array(img_nobg)
    transparent_px = int((arr[:, :, 3] == 0).sum())
    total_px = img.width * img.height
    print(f"OK  ({transparent_px} px removed, {transparent_px*100//total_px}% of image)")

    if save_steps:
        p = out_dir / f"{stem}_01_nobg.png"
        img_nobg.save(p)
        print(f"         -> Saved: {p.name}")

    # Step 3: 检测像素网格
    print(f"[3/4] Detecting pixel grid (max_size={max_pixel_size})...", end="", flush=True)
    pixel_size, offset_x, offset_y, scores = detect_pixel_grid(
        img_nobg, max_pixel_size=max_pixel_size
    )

    print(f"\n  +-- Grid Analysis Result --------------------------------+")
    print(f"  |  Logical pixel size : {pixel_size} x {pixel_size} screen pixels")
    print(f"  |  Grid start offset  : X={offset_x}, Y={offset_y}")
    print(f"  |  Top-5 candidates (higher harmony = better alignment):")

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    for rank, (sz, sc) in enumerate(sorted_scores[:5], 1):
        marker = " <-- BEST" if sz == pixel_size else ""
        print(f"  |    #{rank}  {sz:2d}px  harmony={sc}{marker}")
    print(f"  +--------------------------------------------------------+")

    # Step 4: 纯色度检测 & 生成修复版
    print(f"\n[4/4] Checking purity & generating pixel-perfect version...", end=" ", flush=True)
    purity_map, purity_ratio, impure_count, total_count = check_purity(
        img_nobg, pixel_size, offset_x, offset_y, purity_threshold
    )
    img_perfect, img_native = generate_pixel_perfect(
        img_nobg, pixel_size, offset_x, offset_y
    )
    img_diag = generate_diagnostic(img_nobg, purity_map, pixel_size,
                                   offset_x, offset_y, purity_ratio)
    print("OK")

    perfect_path = out_dir / f"{stem}_02_pixel_perfect.png"
    native_path  = out_dir / f"{stem}_03_native.png"
    diag_path    = out_dir / f"{stem}_04_diagnostic.png"
    img_perfect.save(perfect_path)
    img_native.save(native_path)
    img_diag.save(diag_path)

    # 评级
    grade = ("A+ Perfect" if purity_ratio >= 0.97 else
             "A  Excellent" if purity_ratio >= 0.90 else
             "B  Good" if purity_ratio >= 0.75 else
             "C  Fair" if purity_ratio >= 0.50 else
             "D  Needs heavy repair")

    pure_count = total_count - impure_count

    print(f"\n{'-' * 60}")
    print(f"  Purity Report")
    print(f"{'-' * 60}")
    print(f"  Total cells   : {total_count}")
    print(f"  Pure cells    : {pure_count}  ({purity_ratio*100:.1f}%)")
    print(f"  Impure cells  : {impure_count}  ({(1-purity_ratio)*100:.1f}%)")
    print(f"  Grade         : {grade}")
    print(f"{'-' * 60}")
    native_size = img_native.size
    print(f"\n  Output files:")
    if save_steps:
        print(f"  [1] No-BG        : {stem}_01_nobg.png")
    print(f"  [2] Pixel-Perfect: {perfect_path.name}  ({img_perfect.width}x{img_perfect.height})")
    print(f"  [3] Native px    : {native_path.name}  ({native_size[0]}x{native_size[1]} logical px)")
    print(f"  [4] Diagnostic   : {diag_path.name}")
    print(f"{sep}\n")

    return {
        "input": str(src),
        "size": img.size,
        "pixel_size": pixel_size,
        "offset": (offset_x, offset_y),
        "purity_ratio": purity_ratio,
        "total_cells": total_count,
        "impure_cells": impure_count,
        "pure_cells": pure_count,
        "grade": grade,
        "native_resolution": native_size,
        "outputs": {
            "nobg": str(out_dir / f"{stem}_01_nobg.png") if save_steps else None,
            "pixel_perfect": str(perfect_path),
            "native": str(native_path),
            "diagnostic": str(diag_path),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. 局部 Pixel 检测与网格对齐可视化
# ──────────────────────────────────────────────────────────────────────────────

def _variance_map(rgb: np.ndarray, valid: np.ndarray, win: int) -> np.ndarray:
    """
    计算每个像素位置的局部颜色方差（3通道之和），使用均匀滑动窗口。
    非透明区域外设为极大值（排除）。
    """
    from scipy.ndimage import uniform_filter
    var = np.zeros(rgb.shape[:2], dtype=np.float32)
    for c in range(3):
        ch = rgb[:, :, c].astype(np.float32)
        sz = win * 2 + 1
        m1 = uniform_filter(ch, sz, mode="nearest")
        m2 = uniform_filter(ch ** 2, sz, mode="nearest")
        var += np.maximum(0.0, m2 - m1 ** 2)
    var[~valid] = 1e9
    return var


def find_local_pixels(img_rgba: Image.Image,
                      pixel_size: int,
                      var_threshold: float = 400.0) -> list[dict]:
    """
    在图像中搜索每一个"局部 pixel"：颜色均匀、面积约为 pixel_size² 的小区域。

    算法：
      1. 计算方差图（小窗口内各通道方差之和）
      2. 找方差图的局部极小值（各 pixel 的"最纯核心"）
      3. 以每个极小值为种子，BFS 扩展到相邻颜色相近的像素，
         直到区域面积达到 pixel_size² 或颜色偏差超限
      4. 过滤掉面积过小的片段

    返回：[{'centroid':(cx,cy), 'bbox':(x0,y0,x1,y1),
             'color':(r,g,b), 'area':int, 'var':float}, ...]
    """
    from scipy.ndimage import minimum_filter, label as scipy_label

    data = np.array(img_rgba)
    valid = data[:, :, 3] > 0 if data.shape[2] == 4 else np.ones(data.shape[:2], dtype=bool)
    rgb = data[:, :, :3].astype(np.float32)
    h, w = rgb.shape[:2]

    # ── 方差图 ──────────────────────────────────────────────────
    win = max(2, pixel_size // 4)
    var = _variance_map(rgb, valid, win)

    # ── 局部极小值（每个 pixel 最"纯"的核心点）────────────────
    # 搜索半径 = pixel_size // 2，确保不同 pixel 的极小值间距合理
    sep = max(pixel_size // 2, 3)
    local_min = minimum_filter(var, size=sep * 2 + 1, mode="constant", cval=1e9)
    seed_mask = (var <= local_min + 1e-3) & valid & (var < var_threshold)

    # ── BFS 从每个极小值扩展区域 ─────────────────────────────
    # 先标记连通域种子
    seed_labeled, n_seeds = scipy_label(seed_mask)
    min_area = max(4, int(pixel_size * pixel_size * 0.12))
    target_area = pixel_size * pixel_size
    color_tol = 55.0    # BFS 扩展时，与种子色差不超过此值

    local_pixels: list[dict] = []
    claimed = np.zeros((h, w), dtype=np.int32)  # 0=未认领, >0=已认领 by seed id

    from collections import deque

    for sid in range(1, n_seeds + 1):
        ys_s, xs_s = np.where(seed_labeled == sid)
        if len(ys_s) == 0:
            continue

        # 种子区域颜色中值
        seed_color = np.median(rgb[ys_s, xs_s], axis=0)
        cy0 = float(ys_s.mean())
        cx0 = float(xs_s.mean())

        # BFS 扩展
        visited = np.zeros((h, w), dtype=bool)
        queue = deque()
        region_xs, region_ys = [], []

        for y, x in zip(ys_s.tolist(), xs_s.tolist()):
            visited[y, x] = True
            queue.append((y, x))
            region_ys.append(y)
            region_xs.append(x)

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue and len(region_xs) < target_area * 2:
            cy, cx = queue.popleft()
            for dy, dx in dirs:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if visited[ny, nx] or not valid[ny, nx] or claimed[ny, nx] != 0:
                    continue
                diff = float(np.max(np.abs(rgb[ny, nx] - seed_color)))
                if diff > color_tol:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))
                region_xs.append(nx)
                region_ys.append(ny)

        area = len(region_xs)
        if area < min_area:
            continue

        # 标记已认领
        for y, x in zip(region_ys, region_xs):
            claimed[y, x] = sid

        rys = np.array(region_ys)
        rxs = np.array(region_xs)
        centroid = (float(rxs.mean()), float(rys.mean()))
        bbox = (int(rxs.min()), int(rys.min()), int(rxs.max()) + 1, int(rys.max()) + 1)
        color = tuple(np.median(rgb[rys, rxs], axis=0).astype(np.uint8))
        local_var = float(np.mean(var[rys, rxs]))

        local_pixels.append({
            "centroid": centroid,
            "bbox": bbox,
            "color": color,
            "area": area,
            "var": local_var,
            "seed_id": sid,
        })

    return local_pixels


def align_local_pixels_to_grid(local_pixels: list[dict],
                                pixel_size: int,
                                img_size: tuple[int, int]) -> tuple[int, int, dict]:
    """
    在全局偏移空间 (0~pixel_size-1)² 中枚举，找到使"冲突最少"的对齐偏移。

    每个局部 pixel 按其重心就近归入某个格子 (gx, gy)。
    冲突 = 多个局部 pixel 归入同一格子。

    返回：(offset_x, offset_y, cell_map)
      cell_map: {(gx,gy): [local_pixel_dict, ...]}
    """
    best_ox, best_oy = 0, 0
    best_conflicts = len(local_pixels) + 1

    for oy in range(pixel_size):
        for ox in range(pixel_size):
            counts: Counter = Counter()
            for lp in local_pixels:
                cx, cy = lp["centroid"]
                gx = int(round((cx - ox - pixel_size / 2) / pixel_size))
                gy = int(round((cy - oy - pixel_size / 2) / pixel_size))
                counts[(gx, gy)] += 1
            conflicts = sum(1 for v in counts.values() if v > 1)
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_ox, best_oy = ox, oy

    # 用最优偏移建立 cell_map
    cell_map: dict[tuple[int, int], list] = {}
    for lp in local_pixels:
        cx, cy = lp["centroid"]
        gx = int(round((cx - best_ox - pixel_size / 2) / pixel_size))
        gy = int(round((cy - best_oy - pixel_size / 2) / pixel_size))
        lp["grid_cell"] = (gx, gy)
        cell_map.setdefault((gx, gy), []).append(lp)

    return best_ox, best_oy, cell_map


def generate_local_grid_viz(img_rgba: Image.Image,
                             local_pixels: list[dict],
                             cell_map: dict,
                             pixel_size: int,
                             offset_x: int,
                             offset_y: int) -> Image.Image:
    """
    可视化每个局部 pixel 与全局网格的对齐情况。

    图例：
      绿色边框  = 1:1 对齐（该格只有一个局部 pixel）
      橙色边框  = 冲突（多个局部 pixel 映射到同一格）
      灰色虚框  = 空格（有邻格内容，但本格无局部 pixel 命中）
      白色细线  = 全局网格线
      半透明蓝点 = 局部 pixel 真实重心位置
    """
    h, w = img_rgba.height, img_rgba.width

    # 底层：深灰背景（让颜色块更突出）
    base = Image.new("RGBA", (w, h), (40, 40, 45, 255))

    # 层0：原图（低不透明度，仅作轮廓参考）
    ref_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if img_rgba.mode == "RGBA":
        alpha_ch = img_rgba.split()[3]
        ref_layer.paste(img_rgba, mask=alpha_ch)
    else:
        ref_layer.paste(img_rgba)
    # 降低原图透明度
    ref_arr = np.array(ref_layer)
    ref_arr[:, :, 3] = (ref_arr[:, :, 3].astype(np.float32) * 0.25).astype(np.uint8)
    base = Image.alpha_composite(base, Image.fromarray(ref_arr))

    # 层1：每个局部 pixel 的实际 bbox 位置（纯色填充 + 彩色边框）
    color_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cd = ImageDraw.Draw(color_layer)

    for lp in local_pixels:
        bx0, by0, bx1, by1 = lp["bbox"]
        r, g, b = lp["color"]
        cell = lp["grid_cell"]
        n_in_cell = len(cell_map.get(cell, []))
        if n_in_cell == 1:
            border = (40, 220, 40, 255)      # 绿：正常对齐
        else:
            border = (255, 140, 0, 255)      # 橙：冲突

        # 填色：接近不透明，让色块清晰可见
        cd.rectangle([bx0, by0, bx1 - 1, by1 - 1],
                     fill=(r, g, b, 210),
                     outline=border, width=2)

    base = Image.alpha_composite(base, color_layer)

    # 层2：全局网格线 + 空格标记
    grid_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid_layer)

    # 确定有内容的格子集合（邻格非空）
    occupied_cells = set(cell_map.keys())

    def neighbors_occupied(gx, gy):
        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            if (gx + ddx, gy + ddy) in occupied_cells:
                return True
        return False

    # 绘制邻近空格（蓝灰色虚框 = 这些是"漏网"的空格）
    grid_cols = (w - offset_x) // pixel_size
    grid_rows = (h - offset_y) // pixel_size
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            if (gx, gy) not in occupied_cells and neighbors_occupied(gx, gy):
                cx = offset_x + gx * pixel_size
                cy = offset_y + gy * pixel_size
                # 画四角小方块（虚线效果）
                c = (130, 160, 230, 160)
                s = pixel_size - 1
                for (xx, yy) in [(cx, cy), (cx + s - 2, cy),
                                  (cx, cy + s - 2), (cx + s - 2, cy + s - 2)]:
                    gd.rectangle([xx, yy, xx + 1, yy + 1], fill=c)
                # 细边框
                gd.rectangle([cx, cy, cx + s, cy + s],
                             outline=(100, 130, 200, 80), width=1)

    # 全局网格线（半透明白线）
    for x in range(offset_x, w, pixel_size):
        gd.line([(x, 0), (x, h - 1)], fill=(255, 255, 255, 55), width=1)
    for y in range(offset_y, h, pixel_size):
        gd.line([(0, y), (w - 1, y)], fill=(255, 255, 255, 55), width=1)

    base = Image.alpha_composite(base, grid_layer)

    # 层3：局部 pixel 真实重心（小蓝点）
    dot_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dot_layer)
    for lp in local_pixels:
        cx, cy = lp["centroid"]
        r = 1
        dd.ellipse([cx - r, cy - r, cx + r, cy + r],
                   fill=(80, 160, 255, 220))
    base = Image.alpha_composite(base, dot_layer)

    # 信息栏
    info_h = 64
    canvas = Image.new("RGBA", (w, h + info_h), (22, 22, 26, 255))
    canvas.paste(base, (0, 0))
    id_draw = ImageDraw.Draw(canvas)

    conflicts = sum(1 for v in cell_map.values() if len(v) > 1)
    empty_neighbors = sum(
        1 for gy in range(grid_rows)
        for gx in range(grid_cols)
        if (gx, gy) not in occupied_cells and neighbors_occupied(gx, gy)
    )
    text1 = (f"Local pixels found: {len(local_pixels)}  |  "
             f"Grid: {pixel_size}px  Offset: ({offset_x},{offset_y})")
    text2 = (f"Conflicts (orange): {conflicts}  |  "
             f"Empty neighbor cells (gray dots): {empty_neighbors}  |  "
             f"Green=aligned  Blue dot=centroid")
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except Exception:
        font = ImageFont.load_default()

    id_draw.text((10, h + 6), text1, fill=(200, 230, 255, 255), font=font)
    id_draw.text((10, h + 32), text2, fill=(180, 220, 180, 255), font=font)

    return canvas


def run_local_grid_analysis(input_path: str,
                             pixel_size: int | None = None,
                             output_dir: str | None = None,
                             white_threshold: int = 235,
                             max_pixel_size: int = 32) -> str:
    """
    完整的局部 pixel 检测 + 网格对齐可视化流程。
    返回输出图片路径。
    """
    src = Path(input_path)
    out_dir = Path(output_dir) if output_dir else src.parent
    stem = src.stem
    sep = "=" * 60

    print(f"\n{sep}")
    print("  Local Pixel Grid Analysis")
    print(f"{sep}")

    # 加载 & 去白底
    print("[1/4] Load & remove background...", end=" ", flush=True)
    img = Image.open(src).convert("RGB")
    img_nobg = remove_white_background(img, white_threshold)
    print(f"OK  ({img.width}x{img.height})")

    # 检测像素尺寸（或使用传入值）
    if pixel_size is None:
        print("[2/4] Detecting pixel size...", end="", flush=True)
        pixel_size, ox, oy, _ = detect_pixel_grid(img_nobg, max_pixel_size)
    else:
        print(f"[2/4] Using provided pixel_size={pixel_size}, finding offset...", end="", flush=True)
        data = np.array(img_nobg)
        valid = data[:, :, 3] > 0
        rgb = data[:, :, :3]
        best_s, ox, oy = float("inf"), 0, 0
        for oy_ in range(pixel_size):
            for ox_ in range(pixel_size):
                s = _gradient_score_fast(rgb, valid, pixel_size, oy_, ox_)
                if s < best_s:
                    best_s, ox, oy = s, ox_, oy_
        print(f" OK  offset=({ox},{oy})")

    print(f"\n  Pixel size : {pixel_size}px  |  Global offset: ({ox},{oy})")

    # 查找局部 pixels
    print(f"\n[3/4] Finding local pixels...", end=" ", flush=True)
    local_pixels = find_local_pixels(img_nobg, pixel_size)
    print(f"Found {len(local_pixels)}")

    # 网格对齐
    print(f"[4/4] Aligning to grid & generating visualization...", end=" ", flush=True)
    grid_ox, grid_oy, cell_map = align_local_pixels_to_grid(local_pixels, pixel_size, img.size)
    conflicts = sum(1 for v in cell_map.values() if len(v) > 1)

    viz = generate_local_grid_viz(img_nobg, local_pixels, cell_map,
                                  pixel_size, grid_ox, grid_oy)
    out_path = out_dir / f"{stem}_local_grid.png"
    viz.save(out_path)
    print("OK")

    print(f"\n  Grid offset (from local alignment): ({grid_ox}, {grid_oy})")
    print(f"  Conflicts : {conflicts}")
    print(f"  Output    : {out_path.name}")
    print(f"{sep}\n")

    return str(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pixel Perfect Checker -- AI pixel art analysis & repair tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input image path")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: same as input)")
    p.add_argument("--white-threshold", type=int, default=235, metavar="N",
                   help="White removal threshold 0-255, lower=stricter (default: 235)")
    p.add_argument("--max-pixel-size", type=int, default=32, metavar="N",
                   help="Max logical pixel size to test (default: 32)")
    p.add_argument("--purity-threshold", type=int, default=30, metavar="N",
                   help="Max channel diff within a cell to be 'pure' (default: 30)")
    p.add_argument("--pixel-size", type=int, default=None, metavar="N",
                   help="Manually specify logical pixel size (skip auto-detection)")
    p.add_argument("--no-steps", action="store_true",
                   help="Skip saving intermediate no-bg file")
    p.add_argument("--local-grid", action="store_true",
                   help="Run local-pixel grid alignment analysis (outputs *_local_grid.png)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.local_grid:
            run_local_grid_analysis(
                input_path=args.input,
                pixel_size=args.pixel_size,
                output_dir=args.output_dir,
                white_threshold=args.white_threshold,
                max_pixel_size=args.max_pixel_size,
            )
        else:
            process_image(
                input_path=args.input,
                white_threshold=args.white_threshold,
                max_pixel_size=args.max_pixel_size,
                purity_threshold=args.purity_threshold,
                output_dir=args.output_dir,
                save_steps=not args.no_steps,
            )
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nFailed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
