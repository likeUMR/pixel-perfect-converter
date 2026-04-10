#!/usr/bin/env python3
"""
Algorithm 2: 分块网格对齐 (Block Grid)
=========================================
核心思路：
  对内容区域尝试多种切分策略（1×1 / 2×2 / 3×3 / 4×4 等）
  → 对每块独立做网格检测（梯度比率法）
  → 综合评分：块间 pixel_size 一致性 + 块内梯度比分
  → 选最优切分方案
  → 拼接各块局部网格，生成 pixel-perfect 输出

每步保存一张中间图像。

用法：
  python run.py <输入图片> [--pixel-size N]
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import (remove_white_background, get_content_bbox,
                           detect_pixel_size, save_step,
                           make_base_canvas, draw_grid, _get_font)


# ──────────────────────────────────────────────────────────────────────────────
# 局部网格检测（梯度比率法，针对图像切片）
# ──────────────────────────────────────────────────────────────────────────────

def gradient_score_local(rgb: np.ndarray, valid: np.ndarray,
                          size: int, oy: int, ox: int) -> float:
    """格内/格边界梯度比率，越小越对齐。"""
    rgb_f = rgb.astype(np.float32)
    h, w = rgb_f.shape[:2]

    hd = np.sum(np.abs(rgb_f[:, 1:] - rgb_f[:, :-1]), axis=2)
    hv = valid[:, 1:] & valid[:, :-1]
    hb = ((np.arange(w - 1) - ox + 1) % size == 0)[np.newaxis, :]
    hi = hd[hv & ~hb]; hbv = hd[hv & hb]

    vd = np.sum(np.abs(rgb_f[1:] - rgb_f[:-1]), axis=2)
    vv = valid[1:] & valid[:-1]
    vb = ((np.arange(h - 1) - oy + 1) % size == 0)[:, np.newaxis]
    vi = vd[vv & ~vb]; vbv = vd[vv & vb]

    interior = np.concatenate([hi, vi])
    boundary = np.concatenate([hbv, vbv])
    if len(interior) == 0 or len(boundary) == 0:
        return float("inf")
    return float(np.mean(interior)) / (float(np.mean(boundary)) + 1e-6)


def detect_local_grid(rgb: np.ndarray, valid: np.ndarray,
                       hint_size: int,
                       search_range: int = 3) -> tuple[int, int, int, float]:
    """
    在 hint_size ± search_range 范围内搜索最佳 (pixel_size, offset_x, offset_y)。
    返回 (pixel_size, ox, oy, best_score)。
    """
    best_score = float("inf")
    best_ps, best_ox, best_oy = hint_size, 0, 0

    for ps in range(max(2, hint_size - search_range),
                    hint_size + search_range + 1):
        for oy in range(ps):
            for ox in range(ps):
                s = gradient_score_local(rgb, valid, ps, oy, ox)
                if s < best_score:
                    best_score = s
                    best_ps, best_ox, best_oy = ps, ox, oy

    return best_ps, best_ox, best_oy, best_score


# ──────────────────────────────────────────────────────────────────────────────
# 分块策略定义与评分
# ──────────────────────────────────────────────────────────────────────────────

PARTITION_STRATEGIES = [
    (1, 1),   # 整体
    (2, 2),
    (3, 3),
    (4, 4),
    (2, 3),
    (3, 2),
    (4, 3),
    (3, 4),
    (2, 4),
    (4, 2),
]


def evaluate_partition(img_rgba: Image.Image,
                        valid: np.ndarray,
                        rgb: np.ndarray,
                        bbox: tuple[int, int, int, int],
                        rows: int, cols: int,
                        hint_size: int) -> dict:
    """
    对给定切分方案 (rows×cols) 运行局部网格检测，返回评分结果。

    评分由两部分组成：
      consistency: pixel_size 在各块间的一致性（越高越好）
      quality:     各块内梯度比率的平均值（越低越好）
    最终得分 = consistency - quality * 10（越高越好）
    """
    bx0, by0, bx1, by1 = bbox
    cw, ch = bx1 - bx0, by1 - by0
    bw, bh = cw // cols, ch // rows

    if bw < hint_size * 2 or bh < hint_size * 2:
        return None  # 块太小，跳过

    block_results = []
    for r in range(rows):
        for c in range(cols):
            x0 = bx0 + c * bw
            y0 = by0 + r * bh
            x1 = bx0 + (c + 1) * bw if c < cols - 1 else bx1
            y1 = by0 + (r + 1) * bh if r < rows - 1 else by1

            rgb_b = rgb[y0:y1, x0:x1]
            valid_b = valid[y0:y1, x0:x1]

            if valid_b.sum() < (x1 - x0) * (y1 - y0) * 0.05:
                continue  # 该块几乎是透明背景，跳过

            ps, ox, oy, score = detect_local_grid(rgb_b, valid_b, hint_size)
            block_results.append({
                "rect": (x0, y0, x1, y1),
                "ps": ps, "ox": ox + x0, "oy": oy + y0,
                "score": score,
                "row": r, "col": c,
            })

    if not block_results:
        return None

    sizes = [b["ps"] for b in block_results]
    scores = [b["score"] for b in block_results]

    mean_ps = float(np.mean(sizes))
    std_ps = float(np.std(sizes))
    consistency = 1.0 / (1.0 + std_ps)          # 1 = 完全一致
    quality = float(np.mean(scores))              # 低 = 更好

    final_score = consistency * 10 - quality * 8

    return {
        "rows": rows, "cols": cols,
        "blocks": block_results,
        "mean_ps": mean_ps, "std_ps": std_ps,
        "consistency": consistency,
        "quality": quality,
        "final_score": final_score,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────────────────────

def draw_partition_overview(img_rgba: Image.Image, results: list[dict],
                             best_idx: int) -> Image.Image:
    """绘制所有切分方案的评分概览（小缩略图网格）。"""
    thumb_w, thumb_h = 200, 180
    ncols = 5
    nrows = (len(results) + ncols - 1) // ncols
    W = thumb_w * ncols
    H = thumb_h * nrows
    canvas = Image.new("RGBA", (W, H), (28, 28, 34, 255))
    font = _get_font(13)

    base_small = img_rgba.convert("RGBA")

    for idx, res in enumerate(results):
        r_idx = idx // ncols
        c_idx = idx % ncols
        ox_thumb = c_idx * thumb_w
        oy_thumb = r_idx * thumb_h

        # 缩放底图
        ratio = min(thumb_w / base_small.width, (thumb_h - 30) / base_small.height)
        tw = int(base_small.width * ratio)
        th = int(base_small.height * ratio)
        thumb = base_small.resize((tw, th), Image.NEAREST)
        tile = Image.new("RGBA", (thumb_w, thumb_h),
                         (50, 80, 50, 255) if idx == best_idx else (38, 38, 48, 255))
        tile.paste(thumb, ((thumb_w - tw) // 2, 0), thumb.split()[3])

        d = ImageDraw.Draw(tile)
        label = (f"{res['rows']}×{res['cols']}  "
                 f"ps={res['mean_ps']:.1f}±{res['std_ps']:.1f}\n"
                 f"score={res['final_score']:.2f}")
        if idx == best_idx:
            label += "  ★BEST"
        d.text((4, th + 2), label, fill=(230, 230, 200, 255), font=font)

        # 绘制块分割线
        for blk in res["blocks"]:
            x0b, y0b, x1b, y1b = blk["rect"]
            sx0 = int(x0b * ratio) + (thumb_w - tw) // 2
            sy0 = int(y0b * ratio)
            sx1 = int(x1b * ratio) + (thumb_w - tw) // 2
            sy1 = int(y1b * ratio)
            d.rectangle([sx0, sy0, sx1, sy1], outline=(100, 200, 100, 180), width=1)

        canvas.paste(tile, (ox_thumb, oy_thumb))

    return canvas


def draw_local_grids(img_rgba: Image.Image, result: dict) -> Image.Image:
    """在图上绘制每个块的局部网格线（不同颜色区分各块）。"""
    w, h = img_rgba.size
    base = make_base_canvas(img_rgba)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    block_colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (200, 150, 100), (150, 200, 100), (100, 150, 200),
        (220, 100, 150), (100, 220, 150), (150, 100, 220),
    ]

    for i, blk in enumerate(result["blocks"]):
        x0b, y0b, x1b, y1b = blk["rect"]
        ps = blk["ps"]
        ox = blk["ox"] - x0b   # 转回局部坐标的偏移
        oy_local = blk["oy"] - y0b
        clr = block_colors[i % len(block_colors)]

        # 块边框
        od.rectangle([x0b, y0b, x1b - 1, y1b - 1],
                     outline=clr + (200,), width=2)

        # 局部网格线（只在块内）
        grid_alpha = 60
        x = x0b + (ox % ps)
        while x <= x1b:
            od.line([(x, y0b), (x, y1b)], fill=clr + (grid_alpha,), width=1)
            x += ps
        y = y0b + (oy_local % ps)
        while y <= y1b:
            od.line([(x0b, y), (x1b, y)], fill=clr + (grid_alpha,), width=1)
            y += ps

        # 块角落标注
        od.text((x0b + 3, y0b + 3),
                f"{ps}px ({ox%ps},{oy_local%ps})",
                fill=clr + (230,), font=_get_font(12))

    result_img = Image.alpha_composite(base, overlay)
    return result_img


def render_pixel_perfect(rgb: np.ndarray, valid: np.ndarray,
                          result: dict, img_w: int, img_h: int) -> Image.Image:
    """
    按各块的局部网格渲染 pixel-perfect 图像。
    各块拼接时，重叠区域取中值。
    """
    accum = np.zeros((img_h, img_w, 3), dtype=np.float32)
    count = np.zeros((img_h, img_w), dtype=np.int32)

    for blk in result["blocks"]:
        x0b, y0b, x1b, y1b = blk["rect"]
        ps = blk["ps"]
        # 局部偏移（相对于块左上角）
        lox = (blk["ox"] - x0b) % ps
        loy = (blk["oy"] - y0b) % ps

        for cy in range(y0b + loy, y1b, ps):
            for cx in range(x0b + lox, x1b, ps):
                cy2 = min(cy + ps, y1b, img_h)
                cx2 = min(cx + ps, x1b, img_w)
                cell = rgb[cy:cy2, cx:cx2]
                cell_v = valid[cy:cy2, cx:cx2]
                pixels = cell[cell_v]
                if len(pixels) < 2:
                    continue
                color = np.median(pixels, axis=0)
                accum[cy:cy2, cx:cx2] += color
                count[cy:cy2, cx:cx2] += 1

    result_arr = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    mask = count > 0
    result_arr[mask, :3] = np.clip(accum[mask] / count[mask, np.newaxis], 0, 255).astype(np.uint8)
    result_arr[mask, 3] = 255
    return Image.fromarray(result_arr, "RGBA")


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    src = Path(args.input)
    OUT = Path(__file__).parent / "steps"
    OUT.mkdir(exist_ok=True)
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  Algorithm 2: Block Grid (分块网格对齐)")
    print(f"  Input: {src}")
    print(f"{sep}")

    # ── Step 1: 去白底 ────────────────────────────────────────────────────────
    print("\n[1/6] Remove white background...")
    img = Image.open(src).convert("RGB")
    img_nobg = remove_white_background(img, threshold=235)
    save_step(img_nobg, OUT, "01_nobg", "Step1: White background removed")

    data = np.array(img_nobg)
    valid = data[:, :, 3] > 0
    rgb = data[:, :, :3]
    h, w = rgb.shape[:2]
    bbox = get_content_bbox(valid, padding=10)
    bx0, by0, bx1, by1 = bbox
    print(f"  Content: ({bx0},{by0})→({bx1},{by1})  size={bx1-bx0}×{by1-by0}")

    # ── Step 2: 检测全局 pixel_size ───────────────────────────────────────────
    if args.pixel_size:
        hint_ps = args.pixel_size
        print(f"\n[2/6] Using specified pixel_size={hint_ps}")
    else:
        print("\n[2/6] Detecting global pixel size...", end=" ", flush=True)
        hint_ps, _, _ = detect_pixel_size(img_nobg, verbose=True)
        print(f"-> {hint_ps}px")

    # ── Step 3: 尝试所有切分策略并评分 ──────────────────────────────────────
    print(f"\n[3/6] Evaluating {len(PARTITION_STRATEGIES)} partition strategies...")
    all_results = []
    for rows, cols in PARTITION_STRATEGIES:
        res = evaluate_partition(img_nobg, valid, rgb, bbox, rows, cols, hint_ps)
        if res is not None:
            all_results.append(res)
            print(f"  {rows}×{cols}: ps={res['mean_ps']:.1f}±{res['std_ps']:.1f}  "
                  f"consistency={res['consistency']:.3f}  "
                  f"quality={res['quality']:.4f}  "
                  f"score={res['final_score']:.3f}")
        else:
            print(f"  {rows}×{cols}: skipped (insufficient data)")

    if not all_results:
        print("  ERROR: No valid partition found!")
        return

    # 综合评分排序
    all_results.sort(key=lambda x: -x["final_score"])
    best = all_results[0]
    best_idx_in_orig = next(i for i, r in enumerate(all_results)
                             if r["rows"] == best["rows"] and r["cols"] == best["cols"])

    print(f"\n  Best partition: {best['rows']}×{best['cols']}  "
          f"score={best['final_score']:.3f}  "
          f"mean_ps={best['mean_ps']:.1f}")

    # 可视化：所有方案概览
    overview = draw_partition_overview(img_nobg, all_results, 0)  # idx 0 is best after sort
    save_step(overview, OUT, "02_all_partitions",
              f"Step2: {len(all_results)} strategies evaluated. "
              f"Green=best ({best['rows']}×{best['cols']})")

    # ── Step 4: 最佳方案的局部网格 ───────────────────────────────────────────
    print(f"\n[4/6] Best partition: {best['rows']}×{best['cols']}")
    for blk in best["blocks"]:
        print(f"  Block ({blk['row']},{blk['col']}): "
              f"ps={blk['ps']}  offset=({blk['ox']%blk['ps']},{blk['oy']%blk['ps']})  "
              f"score={blk['score']:.4f}")

    local_grid_vis = draw_local_grids(img_nobg, best)
    save_step(local_grid_vis, OUT, "03_local_grids",
              f"Step3: Local grids in best partition ({best['rows']}×{best['cols']}). "
              f"Each color=one block")

    # ── Step 5: 拼接各块，渲染 pixel-perfect ─────────────────────────────────
    print(f"\n[5/6] Rendering pixel-perfect output by stitching blocks...")
    final_img = render_pixel_perfect(rgb, valid, best, w, h)
    save_step(final_img, OUT, "04_pixel_perfect_stitched",
              f"Step4: Pixel-perfect output stitched from {len(best['blocks'])} blocks")

    # ── Step 6: 对比全局单网格 vs 分块网格 ───────────────────────────────────
    print(f"\n[6/6] Comparison: global grid vs best partition...")

    # 全局网格版本（参考）
    global_res = next((r for r in all_results if r["rows"] == 1 and r["cols"] == 1), None)
    if global_res:
        global_img = render_pixel_perfect(rgb, valid, global_res, w, h)
        # 并排比较
        comp_w = w * 2 + 10
        comp = Image.new("RGBA", (comp_w, h + 50), (28, 28, 34, 255))
        comp.paste(global_img, (0, 0))
        comp.paste(final_img, (w + 10, 0))
        cd = ImageDraw.Draw(comp)
        font = _get_font(15)
        cd.text((10, h + 8),
                f"LEFT: Global 1×1 (score={global_res['final_score']:.2f})  |  "
                f"RIGHT: Best {best['rows']}×{best['cols']} (score={best['final_score']:.2f})",
                fill=(220, 220, 200, 255), font=font)
        save_step(comp, OUT, "05_comparison",
                  "Step5: Left=global single grid, Right=best partition")

    save_step(final_img, OUT, "06_final",
              f"Algorithm2 FINAL: {best['rows']}×{best['cols']} blocks  "
              f"mean_ps={best['mean_ps']:.1f}px")

    print(f"\nDone! All steps saved to: {OUT}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Algorithm 2: Block Grid")
    p.add_argument("input", help="Input image path")
    p.add_argument("--pixel-size", type=int, default=None)
    main(p.parse_args())
