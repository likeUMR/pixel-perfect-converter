#!/usr/bin/env python3
"""
Algorithm 3: 梯度边界线检测 + Hough 式网格拟合
================================================
核心思路：
  图像 → Sobel 梯度图 → 行/列投影 → FFT 找周期（pixel_size）
  → 每个局部水平带/垂直带独立找相位 → 漂移场可视化
  → 用局部相位构建自适应网格 → pixel-perfect 输出

优点：
  - 不需要找"局部像素区域"，直接从边界信号推断
  - 能可视化网格在不同区域的漂移量
  - 对于 AI 生成的渐变边界也有一定鲁棒性

每步保存一张中间图像。

用法：
  python run.py <输入图片> [--pixel-size N]
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import sobel, uniform_filter
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import (remove_white_background, get_content_bbox,
                           detect_pixel_size, save_step,
                           make_base_canvas, draw_grid, _get_font)


# ──────────────────────────────────────────────────────────────────────────────
# 梯度与投影
# ──────────────────────────────────────────────────────────────────────────────

def compute_gradient(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值，透明区域置零。"""
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]).astype(np.float32)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad[~valid] = 0.0
    return grad


def find_period_fft(signal: np.ndarray,
                    min_period: int, max_period: int) -> tuple[int, float]:
    """
    FFT 法从 1D 信号中找主频对应的周期。
    返回 (period, confidence)，confidence ∈ [0,1]。
    """
    if len(signal) < min_period * 2:
        return min_period, 0.0
    F = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal))
    min_f = 1.0 / max_period
    max_f = 1.0 / max(2, min_period)
    mask = (freqs >= min_f) & (freqs <= max_f)
    if not mask.any():
        return min_period, 0.0
    sub = F * mask
    peak_idx = int(np.argmax(sub))
    if freqs[peak_idx] <= 0:
        return min_period, 0.0
    period = int(round(1.0 / freqs[peak_idx]))
    confidence = float(sub[peak_idx] / (F.sum() + 1e-6))
    return max(2, min(max_period, period)), confidence


def find_phase_correlation(signal: np.ndarray, period: int) -> int:
    """
    用与理想梯度模板（delta 函数列）做互相关，找最佳相位。
    返回 phase ∈ [0, period)。
    """
    n = len(signal)
    best_phase, best_val = 0, -1.0
    template = np.zeros(n)
    for phase in range(period):
        xs = np.arange(phase, n, period)
        template[:] = 0
        template[xs] = 1.0
        val = float(np.dot(signal, template))
        if val > best_val:
            best_val = val
            best_phase = phase
    return best_phase


# ──────────────────────────────────────────────────────────────────────────────
# 局部漂移分析
# ──────────────────────────────────────────────────────────────────────────────

def analyze_local_drift(grad: np.ndarray, valid: np.ndarray,
                         pixel_size: int,
                         global_phase_x: int, global_phase_y: int,
                         window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    在滑动窗口中检测局部相位偏移。

    水平漂移（x方向grid偏移随Y变化）：
      对每个 Y 位置的水平带（宽 window*pixel_size），
      投影水平梯度（|gx|）到 X 轴，求局部相位与全局相位之差。

    垂直漂移（y方向grid偏移随X变化）：
      对每个 X 位置的垂直带，投影垂直梯度（|gy|）到 Y 轴，同理。

    返回：
      drift_x: shape (h,) 每行的 X 方向网格相位偏移（像素数）
      drift_y: shape (w,) 每列的 Y 方向网格相位偏移（像素数）
    """
    h, w = grad.shape
    ps = pixel_size
    half_win = window * ps // 2

    # ── 水平漂移：X-向边界相位随 Y 的变化 ──────────────────────────────────
    # 用水平梯度（|gy|：在Y方向颜色变化 = 水平边界位置）
    gray_tmp = grad  # 已经是梯度幅值
    # 按Y方向滑动带，投影到X轴（求每列的梯度和，得到 vertical edge density）
    drift_x = np.full(h, 0.0)
    for cy in range(0, h, max(1, ps // 2)):
        y0 = max(0, cy - half_win)
        y1 = min(h, cy + half_win)
        band = grad[y0:y1, :]
        valid_band = valid[y0:y1, :]
        if valid_band.sum() < ps * 2:
            continue
        proj = band.sum(axis=0)  # project to x-axis: shows vertical boundaries
        phase = find_phase_correlation(proj, ps)
        drift = (phase - global_phase_x + ps // 2) % ps - ps // 2
        ys = range(cy, min(h, cy + max(1, ps // 2)))
        for yy in ys:
            drift_x[yy] = drift

    # ── 垂直漂移：Y-向边界相位随 X 的变化 ──────────────────────────────────
    drift_y = np.full(w, 0.0)
    for cx in range(0, w, max(1, ps // 2)):
        x0 = max(0, cx - half_win)
        x1 = min(w, cx + half_win)
        band = grad[:, x0:x1]
        valid_band = valid[:, x0:x1]
        if valid_band.sum() < ps * 2:
            continue
        proj = band.sum(axis=1)  # project to y-axis: shows horizontal boundaries
        phase = find_phase_correlation(proj, ps)
        drift = (phase - global_phase_y + ps // 2) % ps - ps // 2
        xs = range(cx, min(w, cx + max(1, ps // 2)))
        for xx in xs:
            drift_y[xx] = drift

    return drift_x, drift_y


# ──────────────────────────────────────────────────────────────────────────────
# 可视化辅助
# ──────────────────────────────────────────────────────────────────────────────

def gradient_to_image(grad: np.ndarray) -> Image.Image:
    """将梯度幅值归一化为灰度图像（热力图风格）。"""
    g = grad.copy()
    g = np.clip(g / (np.percentile(g[g > 0], 99) + 1e-6) * 255, 0, 255)
    arr = g.astype(np.uint8)
    return Image.fromarray(arr, "L").convert("RGBA")


def plot_projection_image(signal: np.ndarray, period: int, phase: int,
                           title: str, w: int = 700, h: int = 200) -> Image.Image:
    """将 1D 投影信号绘制为折线图，并叠加检测到的网格线。"""
    img = Image.new("RGBA", (w, h + 50), (28, 28, 34, 255))
    d = ImageDraw.Draw(img)
    n = len(signal)
    sig = signal.copy()
    sig_max = sig.max() if sig.max() > 0 else 1.0
    sig_norm = sig / sig_max

    # 折线
    points = [(int(i / n * w), int((1 - sig_norm[i]) * (h - 10) + 5))
              for i in range(n)]
    if len(points) > 1:
        d.line(points, fill=(100, 200, 255, 200), width=1)

    # 网格线标注
    for x_pos in range(phase, n, period):
        px = int(x_pos / n * w)
        d.line([(px, 0), (px, h)], fill=(255, 200, 50, 120), width=1)

    font = _get_font(14)
    d.text((6, h + 6),
           f"{title}  period={period}  phase={phase}",
           fill=(210, 210, 200, 255), font=font)
    return img


def drift_to_heatmap(drift: np.ndarray, axis: str,
                      img_size: tuple[int, int]) -> Image.Image:
    """
    将 1D 漂移信号可视化为叠加在图像上的彩色热力图条带。
    axis='x': drift 是每行的 X 漂移（竖向颜色条）
    axis='y': drift 是每列的 Y 漂移（横向颜色条）
    """
    w, h = img_size
    arr = np.zeros((h, w, 4), dtype=np.uint8)

    max_d = max(1, np.abs(drift).max())

    if axis == "x":
        # 每行对应一个漂移值：右移=红，左移=蓝
        for y in range(min(h, len(drift))):
            d_val = drift[y] / max_d  # [-1, 1]
            r = int(max(0, d_val) * 255)
            b = int(max(0, -d_val) * 255)
            arr[y, :, 0] = r
            arr[y, :, 2] = b
            arr[y, :, 3] = min(200, int(abs(d_val) * 200))
    else:
        # 每列对应一个漂移值：下移=绿，上移=紫
        for x in range(min(w, len(drift))):
            d_val = drift[x] / max_d
            g_c = int(max(0, d_val) * 255)
            r_c = int(max(0, -d_val) * 255)
            arr[:, x, 1] = g_c
            arr[:, x, 0] = r_c
            arr[:, x, 3] = min(200, int(abs(d_val) * 200))

    return Image.fromarray(arr, "RGBA")


def plot_drift_signal(drift: np.ndarray, title: str,
                       pixel_size: int,
                       w: int = 700, h: int = 150) -> Image.Image:
    """绘制漂移信号折线图。"""
    img = Image.new("RGBA", (w, h + 50), (28, 28, 34, 255))
    d = ImageDraw.Draw(img)
    n = len(drift)
    if n == 0:
        return img
    half = pixel_size / 2
    center_y = h // 2

    # 零线
    d.line([(0, center_y), (w - 1, center_y)], fill=(100, 100, 100, 150), width=1)
    # ±half 辅助线
    for dv in [-half, half]:
        py = int(center_y - dv / half * (h // 2 - 5))
        d.line([(0, py), (w - 1, py)], fill=(80, 80, 120, 100), width=1)

    points = []
    for i in range(n):
        px = int(i / n * w)
        py = int(center_y - drift[i] / (half + 1e-6) * (h // 2 - 5))
        py = max(5, min(h - 5, py))
        points.append((px, py))
    if len(points) > 1:
        d.line(points, fill=(100, 255, 100, 200), width=1)

    font = _get_font(14)
    d.text((6, h + 6), f"{title}  (range: [{drift.min():.1f}, {drift.max():.1f}]px)",
           fill=(210, 210, 200, 255), font=font)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 像素完美输出（自适应相位）
# ──────────────────────────────────────────────────────────────────────────────

def render_adaptive_grid(rgb: np.ndarray, valid: np.ndarray,
                          pixel_size: int,
                          global_ox: int, global_oy: int,
                          drift_x: np.ndarray, drift_y: np.ndarray) -> Image.Image:
    """
    按局部漂移补偿后的网格渲染 pixel-perfect 图像。
    对每个逻辑格 (gx, gy)，根据该格中心位置查询局部漂移，
    调整实际采样位置后取中值色。
    """
    h, w = rgb.shape[:2]
    grid_rows = (h - global_oy) // pixel_size
    grid_cols = (w - global_ox) // pixel_size
    result = np.zeros((h, w, 4), dtype=np.uint8)

    for gy in range(grid_rows):
        for gx in range(grid_cols):
            # 全局格子位置
            x0 = global_ox + gx * pixel_size
            y0 = global_oy + gy * pixel_size
            cx = x0 + pixel_size // 2
            cy = y0 + pixel_size // 2
            if cx >= w or cy >= h:
                continue

            # 局部漂移补偿
            dx = int(round(drift_x[min(cy, h - 1)]))
            dy = int(round(drift_y[min(cx, w - 1)]))
            sx0 = max(0, min(w - pixel_size, x0 + dx))
            sy0 = max(0, min(h - pixel_size, y0 + dy))
            sx1 = sx0 + pixel_size
            sy1 = sy0 + pixel_size

            cell = rgb[sy0:sy1, sx0:sx1]
            cell_v = valid[sy0:sy1, sx0:sx1]
            pixels = cell[cell_v]
            if len(pixels) < 2:
                continue

            color = np.median(pixels, axis=0).astype(np.uint8)
            result[y0:y0 + pixel_size, x0:x0 + pixel_size, :3] = color
            result[y0:y0 + pixel_size, x0:x0 + pixel_size, 3] = 255

    return Image.fromarray(result, "RGBA")


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    src = Path(args.input)
    OUT = Path(__file__).parent / "steps"
    OUT.mkdir(exist_ok=True)
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  Algorithm 3: Gradient + Hough Grid")
    print(f"  Input: {src}")
    print(f"{sep}")

    # ── Step 1: 去白底 ────────────────────────────────────────────────────────
    print("\n[1/8] Remove white background...")
    img = Image.open(src).convert("RGB")
    img_nobg = remove_white_background(img, threshold=235)
    save_step(img_nobg, OUT, "01_nobg", "Step1: White background removed")

    data = np.array(img_nobg)
    valid = data[:, :, 3] > 0
    rgb = data[:, :, :3]
    h, w = rgb.shape[:2]
    bx0, by0, bx1, by1 = get_content_bbox(valid, padding=10)

    # ── Step 2: 梯度图 ────────────────────────────────────────────────────────
    print("\n[2/8] Computing gradient image...")
    grad = compute_gradient(rgb, valid)
    grad_img = gradient_to_image(grad)
    save_step(grad_img, OUT, "02_gradient",
              "Step2: Sobel gradient magnitude (bright=high gradient=pixel boundary)")

    # 二值边缘图
    thresh_val = np.percentile(grad[grad > 0], 75) if (grad > 0).any() else 1
    edges = (grad > thresh_val).astype(np.uint8) * 255
    edge_img = Image.fromarray(edges, "L").convert("RGBA")
    save_step(edge_img, OUT, "03_edges_binary",
              f"Step3: Binary edges (threshold=75th percentile={thresh_val:.1f})")

    # ── Step 3: 行/列投影 ─────────────────────────────────────────────────────
    print("\n[3/8] Computing projections...")
    # 垂直边界的列投影（用于检测 X 方向周期）
    grad_content = grad[by0:by1, bx0:bx1]
    v_proj = grad_content.sum(axis=0)   # 列方向→检测 X-周期
    h_proj = grad_content.sum(axis=1)   # 行方向→检测 Y-周期

    # ── Step 4: FFT 找周期（pixel_size） ─────────────────────────────────────
    print("\n[4/8] FFT period detection...")
    if args.pixel_size:
        pixel_size = args.pixel_size
        confidence_x = confidence_y = 1.0
        print(f"  Using specified pixel_size={pixel_size}")
    else:
        ps_from_x, conf_x = find_period_fft(v_proj, min_period=3, max_period=40)
        ps_from_y, conf_y = find_period_fft(h_proj, min_period=3, max_period=40)
        print(f"  From X-projection: period={ps_from_x} (confidence={conf_x:.3f})")
        print(f"  From Y-projection: period={ps_from_y} (confidence={conf_y:.3f})")
        # 加权选择（取置信度更高的，或用全局检测做校验）
        pixel_size = ps_from_x if conf_x >= conf_y else ps_from_y
        confidence_x, confidence_y = conf_x, conf_y
        # 与全局检测对比
        ps_global, _, _ = detect_pixel_size(img_nobg, verbose=False)
        print(f"  Global harmony detection: {ps_global}px")
        if abs(ps_global - pixel_size) <= 2:
            pixel_size = ps_global  # 倾向全局更稳定的结果
        print(f"  Final pixel_size: {pixel_size}px")

    # 找全局相位
    global_phase_x = find_phase_correlation(v_proj, pixel_size)
    global_phase_y = find_phase_correlation(h_proj, pixel_size)
    # 转换回全图坐标
    global_ox = (bx0 + global_phase_x) % pixel_size
    global_oy = (by0 + global_phase_y) % pixel_size
    print(f"  Global phase: x_phase={global_phase_x}  y_phase={global_phase_y}")
    print(f"  Global offset: ({global_ox}, {global_oy})")

    # 投影图可视化（拼接两张图）
    proj_x_img = plot_projection_image(v_proj, pixel_size, global_phase_x,
                                        "X-projection (vertical boundary density)")
    proj_y_img = plot_projection_image(h_proj, pixel_size, global_phase_y,
                                        "Y-projection (horizontal boundary density)")
    total_h = proj_x_img.height + proj_y_img.height
    proj_combined = Image.new("RGBA", (max(proj_x_img.width, proj_y_img.width), total_h),
                               (28, 28, 34, 255))
    proj_combined.paste(proj_x_img, (0, 0))
    proj_combined.paste(proj_y_img, (0, proj_x_img.height))
    save_step(proj_combined, OUT, "04_projections",
              f"Step4: Gradient projections  pixel_size={pixel_size}px  "
              f"Yellow lines=detected grid")

    # ── Step 5: 局部漂移分析 ─────────────────────────────────────────────────
    print("\n[5/8] Analyzing local drift...")
    drift_x, drift_y = analyze_local_drift(
        grad, valid, pixel_size, global_phase_x, global_phase_y, window=5
    )
    print(f"  X-drift (per row): range=[{drift_x.min():.1f}, {drift_x.max():.1f}]px")
    print(f"  Y-drift (per col): range=[{drift_y.min():.1f}, {drift_y.max():.1f}]px")

    # 漂移信号折线图
    drift_x_plot = plot_drift_signal(drift_x, "X-drift per row (row→drift in px)",
                                      pixel_size)
    drift_y_plot = plot_drift_signal(drift_y, "Y-drift per col (col→drift in px)",
                                      pixel_size)
    drift_plots = Image.new("RGBA",
                             (max(drift_x_plot.width, drift_y_plot.width),
                              drift_x_plot.height + drift_y_plot.height),
                             (28, 28, 34, 255))
    drift_plots.paste(drift_x_plot, (0, 0))
    drift_plots.paste(drift_y_plot, (0, drift_x_plot.height))
    save_step(drift_plots, OUT, "05_drift_signals",
              f"Step5: Local drift  X=[{drift_x.min():.1f},{drift_x.max():.1f}]  "
              f"Y=[{drift_y.min():.1f},{drift_y.max():.1f}]px")

    # ── Step 6: 漂移热力图叠加 ────────────────────────────────────────────────
    print("\n[6/8] Generating drift heatmap overlay...")
    base = make_base_canvas(img_nobg)
    hm_x = drift_to_heatmap(drift_x, "x", (w, h))
    hm_y = drift_to_heatmap(drift_y, "y", (w, h))
    hm_combined = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    hm_combined = Image.alpha_composite(hm_combined, hm_x)
    hm_combined = Image.alpha_composite(hm_combined, hm_y)
    drift_vis = Image.alpha_composite(base, hm_combined)
    save_step(drift_vis, OUT, "06_drift_heatmap",
              "Step6: Drift heatmap  Red=rightward  Blue=leftward  "
              "Green=downward  Purple=upward")

    # ── Step 7: 全局网格叠加 ──────────────────────────────────────────────────
    print("\n[7/8] Generating global grid overlay...")
    grid_base = make_base_canvas(img_nobg)
    gd = ImageDraw.Draw(grid_base)
    draw_grid(gd, w, h, global_ox, global_oy, pixel_size,
              color=(255, 220, 50, 90), line_w=1)
    save_step(grid_base, OUT, "07_global_grid_overlay",
              f"Step7: Global grid overlay  pixel_size={pixel_size}px  "
              f"offset=({global_ox},{global_oy})")

    # ── Step 8: 渲染自适应网格 pixel-perfect ─────────────────────────────────
    print("\n[8/8] Rendering adaptive grid pixel-perfect output...")
    final = render_adaptive_grid(rgb, valid, pixel_size,
                                  global_ox, global_oy, drift_x, drift_y)
    save_step(final, OUT, "08_final",
              f"Algorithm3 FINAL: Adaptive grid  pixel_size={pixel_size}px  "
              f"drift-compensated")

    print(f"\nDone! All steps saved to: {OUT}")
    print(f"pixel_size={pixel_size}  global_offset=({global_ox},{global_oy})")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Algorithm 3: Gradient + Hough")
    p.add_argument("input", help="Input image path")
    p.add_argument("--pixel-size", type=int, default=None)
    main(p.parse_args())
