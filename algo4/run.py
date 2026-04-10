#!/usr/bin/env python3
"""
algo4/run.py — 经典网格识别 + 网格遵循度评估
=============================================

思路：
  给定参考网格图（如 grid_1px_black.png），从中解析 pixel_size 和 offset。
  检测输入图像是否存在可见网格线（AI 是否将网格线画进了图）。
  按格子逐块分析：每个格子的纯色度、颜色、覆盖率。
  计算三项核心指标评估 AI 对网格的遵循程度：
    - CPR  : 格子纯色率（Cell Purity Rate）
    - MCV  : 格内平均方差（Mean Cell Variance，越低越好）
    - BIR  : 边界/内部比（格间色差 / 格内色标准差，越高越好）
    - score: 综合遵循分（0-100）
  生成像素完美输出（取每格中心区域中位色填充）。
  输出可视化：纯色热力图、边界锐度图、综合对比卡。

用法：
  cd algo4
  python run.py <图片路径> [--grid-ref <参考网格路径>] [--pixel-size N]
                           [--out-dir <输出目录>] [--save-steps]
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import remove_white_background, get_content_bbox, _get_font

# ─────────────────────────────────────────────────────────────────────────────
# 默认参数
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_GRID_REF    = Path(__file__).parent.parent / "pixel_grid" / "grid_1px_black.png"
PURITY_THRESH_HARD  = 200     # 格内3通道方差和 < 此值 → "严格纯色"
PURITY_THRESH_SOFT  = 2000    # < 此值 → "宽松纯色"（用于综合分）
BG_THRESH           = 230     # 格子平均亮度 > 此值 → 背景格，不参与指标
DARK_LINE_THRESH    = 80      # 暗像素阈值（均值 RGB < 此值视为暗）
GRID_LINE_COL_RATIO = 0.25    # 一列中暗像素比例 > 此值 → 疑似网格线列
BIR_CAP             = 6.0     # BIR 归一化上限（BIR ≥ BIR_CAP 时得满分）


# ─────────────────────────────────────────────────────────────────────────────
# 1. 解析参考网格 → pixel_size, offset
# ─────────────────────────────────────────────────────────────────────────────

def parse_reference_grid(grid_path: Path) -> tuple[int, int, int]:
    """
    从参考网格图中提取 (pixel_size, offset_x, offset_y)。
    算法：对每一列统计"暗像素"比例，比例高的列是网格线；
    用相邻网格线间距的众数确定 pixel_size，第一条线位置为 offset。
    """
    arr = np.array(Image.open(grid_path).convert("RGB"))
    h, w = arr.shape[:2]

    # 每列：平均灰度低 → 暗 → 网格线候选
    col_mean = arr.mean(axis=2).mean(axis=0)       # (W,)
    row_mean = arr.mean(axis=2).mean(axis=1)       # (H,)

    def find_period_and_offset(signal: np.ndarray, length: int) -> tuple[int, int]:
        dark_cols = np.where(signal < DARK_LINE_THRESH)[0]
        if len(dark_cols) < 2:
            return 8, 0
        gaps = np.diff(dark_cols)
        # 众数间距
        from collections import Counter
        spacing = int(Counter(gaps.tolist()).most_common(1)[0][0])
        offset  = int(dark_cols[0])
        return spacing, offset

    pixel_size_x, offset_x = find_period_and_offset(col_mean, w)
    pixel_size_y, offset_y = find_period_and_offset(row_mean, h)
    # 取两个方向的众数（通常一致）
    pixel_size = int(np.median([pixel_size_x, pixel_size_y]))
    return pixel_size, offset_x, offset_y


# ─────────────────────────────────────────────────────────────────────────────
# 2. 检测图像中是否存在可见网格线
# ─────────────────────────────────────────────────────────────────────────────

def detect_visible_gridlines(rgb: np.ndarray, pixel_size: int,
                              offset_x: int, offset_y: int) -> bool:
    """
    判断图像是否包含可见网格线（AI 将参考网格的黑线绘入图中）。
    方法：比较"预期网格线列"与"非网格列"的平均暗像素比例。
    若网格线列的暗像素显著多于非网格列，说明网格线可见。
    """
    h, w = rgb.shape[:2]
    # 用 DARK_LINE_THRESH（<80）检测暗像素：覆盖纯黑和深灰色网格线
    gray = rgb.mean(axis=2)                     # (H, W)
    dark = gray < DARK_LINE_THRESH

    xs = np.arange(w)
    ys = np.arange(h)
    is_grid_col = ((xs - offset_x) % pixel_size == 0)
    is_grid_row = ((ys - offset_y) % pixel_size == 0)

    col_dark_ratio = dark.mean(axis=0)          # (W,)
    grid_col_ratio    = col_dark_ratio[is_grid_col].mean()
    nongrid_col_ratio = col_dark_ratio[~is_grid_col].mean() if (~is_grid_col).any() else 0

    row_dark_ratio = dark.mean(axis=1)
    grid_row_ratio    = row_dark_ratio[is_grid_row].mean()
    nongrid_row_ratio = row_dark_ratio[~is_grid_row].mean() if (~is_grid_row).any() else 0

    ratio_h = grid_col_ratio / (nongrid_col_ratio + 1e-6)
    ratio_v = grid_row_ratio / (nongrid_row_ratio + 1e-6)
    # 阈值 2.0（比原先 2.5 略宽松，以捕捉深色角色部分遮挡网格线的情况）
    return bool((ratio_h > 2.0) or (ratio_v > 2.0))


# ─────────────────────────────────────────────────────────────────────────────
# 3. 逐格分析
# ─────────────────────────────────────────────────────────────────────────────

def analyze_cells(rgb: np.ndarray,
                  valid: np.ndarray,
                  pixel_size: int,
                  offset_x: int,
                  offset_y: int,
                  grid_margin: int = 1,
                  center_sample: int = 4) -> dict:
    """
    遍历网格的每个格子（由 offset 和 pixel_size 定义），计算：
      - 'median'   : 格子中心 center_sample×center_sample 区域的中位色（用于取色）
      - 'variance' : 完整格内区域（去掉 grid_margin 边距后）的3通道方差之和（用于度量）
      - 'coverage' : 格内有效像素比例 [0,1]
      - 'is_bg'    : 是否背景格（覆盖率低或平均亮度高）

    grid_margin  : 每边跳过的像素数（有可见网格线时为1，否则为0），影响 variance 区域
    center_sample: 中心取色窗口大小（NxN），默认4。取格子正中心的小块作为代表色，
                   可有效规避网格线和边缘走样的干扰。
    """
    h, w = rgb.shape[:2]
    ps = pixel_size
    gm = grid_margin
    cs = center_sample
    # 中心偏移：使 cs×cs 窗口居中于整个 ps×ps 格子
    cs_off_lo = (ps - cs) // 2   # 例如 ps=8, cs=4 → cs_off_lo=2
    cs_off_hi = cs_off_lo + cs   #                   → cs_off_hi=6

    gx_start = 0 if offset_x == 0 else -1
    gy_start = 0 if offset_y == 0 else -1

    cells = {}
    gy = gy_start
    while True:
        y0 = offset_y + gy * ps
        y1 = y0 + ps
        if y0 >= h:
            break

        # ── 完整内部区域（用于 variance 计算）──
        iy0 = max(0, y0 + gm)
        iy1 = min(h, y1 - gm)
        # ── 中心取色区域（用于 median 计算）──
        sy0 = max(0, y0 + cs_off_lo)
        sy1 = min(h, y0 + cs_off_hi)

        if iy0 >= iy1:
            gy += 1
            continue

        gx = gx_start
        while True:
            x0 = offset_x + gx * ps
            x1 = x0 + ps
            if x0 >= w:
                break

            ix0 = max(0, x0 + gm)
            ix1 = min(w, x1 - gm)
            sx0 = max(0, x0 + cs_off_lo)
            sx1 = min(w, x0 + cs_off_hi)

            if ix0 >= ix1:
                gx += 1
                continue

            # ── 完整内部 patch（方差用）──
            patch       = rgb[iy0:iy1, ix0:ix1].astype(np.float32)
            patch_valid = valid[iy0:iy1, ix0:ix1]
            n_valid  = int(patch_valid.sum())
            n_total  = patch.shape[0] * patch.shape[1]
            coverage = n_valid / max(n_total, 1)

            if n_valid == 0:
                is_bg    = True
                median_c = np.array([255, 255, 255], dtype=float)
                variance = 0.0
            else:
                pix_full = patch[patch_valid]
                mean_brightness = float(pix_full.mean())
                is_bg    = (mean_brightness > BG_THRESH) or (coverage < 0.3)
                variance = float(np.var(pix_full, axis=0).sum()) if not is_bg else 0.0

                # ── 中心取色 patch（median 用）──
                if sy0 < sy1 and sx0 < sx1:
                    cpatch       = rgb[sy0:sy1, sx0:sx1].astype(np.float32)
                    cpatch_valid = valid[sy0:sy1, sx0:sx1]
                    cpix = cpatch[cpatch_valid] if cpatch_valid.any() else pix_full
                else:
                    cpix = pix_full
                # 若中心全是背景像素，回退到全区域
                if len(cpix) == 0 or cpix.mean() > BG_THRESH:
                    cpix = pix_full
                median_c = np.median(cpix, axis=0)

            cells[(gy, gx)] = {
                'gy': gy, 'gx': gx,
                'img_y0': iy0, 'img_y1': iy1,
                'img_x0': ix0, 'img_x1': ix1,
                'cell_y0': y0, 'cell_x0': x0,
                'cell_y1': y1, 'cell_x1': x1,
                'sample_y0': sy0, 'sample_x0': sx0,  # 取色区域（调试用）
                'sample_y1': sy1, 'sample_x1': sx1,
                'median': median_c,
                'variance': variance,
                'coverage': coverage,
                'is_bg': is_bg,
            }
            gx += 1

        gy += 1

    return cells


# ─────────────────────────────────────────────────────────────────────────────
# 4. 计算遵循度指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(cells: dict) -> dict:
    """
    计算四项核心指标：

    CPR_hard  : 格内方差 < PURITY_THRESH_HARD 的格子占比（严格纯色率）
    CPR_soft  : 格内方差 < PURITY_THRESH_SOFT 的格子占比（宽松纯色率）
    MCV       : 所有内容格的平均方差（越低越好）
    BIR       : 格间色差均值 / 格内标准差均值（越高越好，理想 >> 1）
    score     : 综合分 0-100
                = CPR_soft * 40 + min(BIR / BIR_CAP, 1) * 60
    """
    content = {k: v for k, v in cells.items() if not v['is_bg']}
    if not content:
        return {'CPR_hard': 0, 'CPR_soft': 0, 'MCV': 0,
                'BIR': 0, 'score': 0, 'n_content': 0, 'n_total': len(cells)}

    variances = np.array([v['variance'] for v in content.values()])
    CPR_hard = float((variances < PURITY_THRESH_HARD).mean())
    CPR_soft = float((variances < PURITY_THRESH_SOFT).mean())
    MCV      = float(variances.mean())

    # BIR: 相邻格对的中位色 L1 差距 / 格内 std 均值
    within_stds = np.sqrt(np.maximum(variances, 0))
    mean_within_std = float(within_stds.mean()) if len(within_stds) else 1.0

    between_diffs = []
    for (gy, gx), cv in content.items():
        for ny, nx in [(gy, gx + 1), (gy + 1, gx)]:
            if (ny, nx) in content:
                diff = float(np.abs(cv['median'] - content[(ny, nx)]['median']).sum())
                between_diffs.append(diff)

    mean_between = float(np.mean(between_diffs)) if between_diffs else 0.0
    BIR = mean_between / (mean_within_std + 1e-6)

    score = CPR_soft * 40 + min(BIR / BIR_CAP, 1.0) * 60

    return {
        'CPR_hard':      CPR_hard,
        'CPR_soft':      CPR_soft,
        'MCV':           MCV,
        'BIR':           BIR,
        'score':         score,
        'n_content':     len(content),
        'n_total':       len(cells),
        'center_sample': 4,   # placeholder; overwritten by run_pipeline
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. 像素完美渲染
# ─────────────────────────────────────────────────────────────────────────────

def render_pixel_perfect(rgb: np.ndarray,
                         valid: np.ndarray,
                         cells: dict,
                         has_grid_lines: bool) -> np.ndarray:
    """
    像素完美重建：以整个 pixel_size×pixel_size 格子为单位，
    用每格的代表色（中心4×4中位色）填满整格。
    背景格填白色。不保留任何网格线——pixel-perfect 输出中
    每个逻辑像素就是一个完整的纯色方块。
    """
    h, w = rgb.shape[:2]
    out = np.ones((h, w, 3), dtype=np.uint8) * 255   # 白色背景

    for (gy, gx), c in cells.items():
        # 使用完整格子坐标（cell_y0/x0），而非内部裁剪坐标（img_y0/x0）
        cy0 = max(0, c['cell_y0'])
        cy1 = min(h, c['cell_y1'])
        cx0 = max(0, c['cell_x0'])
        cx1 = min(w, c['cell_x1'])
        if cy0 >= cy1 or cx0 >= cx1:
            continue
        if c['is_bg']:
            out[cy0:cy1, cx0:cx1] = 255
        else:
            color = np.round(c['median']).astype(np.uint8)
            out[cy0:cy1, cx0:cx1] = color

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5b. 透明背景版（RGBA）
# ─────────────────────────────────────────────────────────────────────────────

def make_transparent(pp_arr: np.ndarray, cells: dict, h: int, w: int,
                     pixel_size: int) -> Image.Image:
    """
    对像素完美输出进行 BFS 去白色背景，返回 RGBA Image。

    策略：
    1. 在【格子粒度】上做 BFS：从四边的白色格（均值亮度 > BG_THRESH）开始，
       向内扩散，标记所有连通的白色格为背景。
    2. 用标记结果填写 alpha 通道（背景格 → alpha=0，内容格 → alpha=255）。

    为什么不直接用 cells.is_bg？
    原图 remove_white_background 的 BFS 对角色铺满画布的图可能失效，
    导致所有格子都被判为内容格，背景无法置透。
    而像素完美输出已是纯色块，格子级 BFS 更稳定。
    """
    from collections import deque

    # ── 收集格子范围 ──
    if not cells:
        return Image.fromarray(
            np.zeros((h, w, 4), dtype=np.uint8), "RGBA")

    gy_vals = [k[0] for k in cells]
    gx_vals = [k[1] for k in cells]
    min_gy, max_gy = min(gy_vals), max(gy_vals)
    min_gx, max_gx = min(gx_vals), max(gx_vals)

    # ── 格子亮度矩阵（格子坐标系）──
    rows = max_gy - min_gy + 1
    cols = max_gx - min_gx + 1
    bright = np.zeros((rows, cols), dtype=np.float32)
    has_cell = np.zeros((rows, cols), dtype=bool)

    for (gy, gx), c in cells.items():
        r, co = gy - min_gy, gx - min_gx
        has_cell[r, co] = True
        # 用 pixel-perfect 输出的格子中心颜色作为亮度依据
        cy0 = max(0, c['cell_y0']); cy1 = min(h, c['cell_y1'])
        cx0 = max(0, c['cell_x0']); cx1 = min(w, c['cell_x1'])
        patch = pp_arr[cy0:cy1, cx0:cx1].astype(np.float32)
        bright[r, co] = patch.mean() if patch.size > 0 else 255.0

    is_white = bright > BG_THRESH   # 亮度高 → 候选背景格

    # ── BFS 从四边白色格向内泛洪 ──
    is_bg_grid = np.zeros((rows, cols), dtype=bool)
    queue = deque()

    def try_enq(r, co):
        if 0 <= r < rows and 0 <= co < cols and has_cell[r, co] \
                and is_white[r, co] and not is_bg_grid[r, co]:
            is_bg_grid[r, co] = True
            queue.append((r, co))

    for co in range(cols):
        try_enq(0, co)
        try_enq(rows - 1, co)
    for r in range(rows):
        try_enq(r, 0)
        try_enq(r, cols - 1)

    while queue:
        r, co = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            try_enq(r + dr, co + dc)

    # ── 写 alpha 通道 ──
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = pp_arr
    out[:, :, 3]  = 0   # 全透明起点

    for (gy, gx), c in cells.items():
        r, co = gy - min_gy, gx - min_gx
        cy0 = max(0, c['cell_y0']); cy1 = min(h, c['cell_y1'])
        cx0 = max(0, c['cell_x0']); cx1 = min(w, c['cell_x1'])
        if cy0 >= cy1 or cx0 >= cx1:
            continue
        if not is_bg_grid[r, co]:
            out[cy0:cy1, cx0:cx1, 3] = 255   # 内容格 → 不透明

    return Image.fromarray(out, "RGBA")


# ─────────────────────────────────────────────────────────────────────────────
# 5c. 迷你像素版（1 逻辑像素 = 1 图像像素）
# ─────────────────────────────────────────────────────────────────────────────

def make_mini_pixel(cells: dict, tp_img: Image.Image,
                    pixel_size: int) -> Image.Image:
    """
    将每个逻辑像素（格子）压缩为 1 个图像像素，生成迷你版。
    透明度与 tp_img（透明背景版）一致：BFS 已正确判断背景格。
    输出尺寸 = (max_gx - min_gx + 1) × (max_gy - min_gy + 1)，
    例如 512×512 图像、pixel_size=8 → 64×64 输出。
    """
    if not cells:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    gy_vals = [k[0] for k in cells]
    gx_vals = [k[1] for k in cells]
    min_gy, max_gy = min(gy_vals), max(gy_vals)
    min_gx, max_gx = min(gx_vals), max(gx_vals)
    h_mini = max_gy - min_gy + 1
    w_mini = max_gx - min_gx + 1

    out  = np.zeros((h_mini, w_mini, 4), dtype=np.uint8)   # 全透明
    tp_a = np.array(tp_img)[:, :, 3]   # 从透明背景版取 alpha 通道

    for (gy, gx), c in cells.items():
        py = gy - min_gy
        px = gx - min_gx
        # 从 tp_img 的格子中心采样 alpha 来判断是否透明
        cy_c = (c['cell_y0'] + c['cell_y1']) // 2
        cx_c = (c['cell_x0'] + c['cell_x1']) // 2
        cy_c = min(max(cy_c, 0), tp_a.shape[0] - 1)
        cx_c = min(max(cx_c, 0), tp_a.shape[1] - 1)
        if tp_a[cy_c, cx_c] > 0:       # 内容格（不透明）
            color = np.round(c['median']).astype(np.uint8)
            out[py, px, :3] = color
            out[py, px, 3]  = 255

    return Image.fromarray(out, "RGBA")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 可视化：纯色热力图
# ─────────────────────────────────────────────────────────────────────────────

def make_purity_heatmap(cells: dict, h: int, w: int) -> Image.Image:
    """
    每格用颜色编码其内部方差（遵循程度）：
      深绿  : variance < PURITY_THRESH_HARD   （严格纯色）
      黄绿  : variance < PURITY_THRESH_SOFT   （宽松纯色）
      橙红  : variance < 5000                 （混浊）
      深红  : variance >= 5000                （高度混浊）
      灰色  : 背景格
    """
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def var_to_color(var: float) -> tuple[int, int, int]:
        if var < PURITY_THRESH_HARD:
            return (40, 200, 80)     # 深绿
        elif var < PURITY_THRESH_SOFT:
            return (180, 220, 60)    # 黄绿
        elif var < 5000:
            return (240, 140, 30)    # 橙
        else:
            return (200, 40, 40)     # 深红

    for (gy, gx), c in cells.items():
        iy0, iy1 = c['img_y0'], c['img_y1']
        ix0, ix1 = c['img_x0'], c['img_x1']
        if c['is_bg']:
            canvas[iy0:iy1, ix0:ix1] = (70, 70, 80)
        else:
            canvas[iy0:iy1, ix0:ix1] = var_to_color(c['variance'])

    return Image.fromarray(canvas, "RGB")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 可视化：边界锐度图
# ─────────────────────────────────────────────────────────────────────────────

def make_boundary_viz(cells: dict, h: int, w: int,
                      pixel_size: int, has_grid_lines: bool) -> Image.Image:
    """
    对每对相邻内容格，在它们之间的边界线上绘制色差强度：
      绿色（diff > 80）: 边界锐利 → 颜色在格线处切换，遵循良好
      黄色（diff 30-80）: 中等
      红色（diff < 30）: 边界模糊 → 内容跨越格线，遵循差

    背景格之间的边界不绘制（灰色）。
    """
    # 先用各格中位色画底图，便于理解空间位置
    base = np.ones((h, w, 3), dtype=np.uint8) * 30   # 深色背景

    # 画各格中位色（半透明感：混入深色）
    for (gy, gx), c in cells.items():
        iy0, iy1 = c['img_y0'], c['img_y1']
        ix0, ix1 = c['img_x0'], c['img_x1']
        if not c['is_bg']:
            color = np.round(c['median'] * 0.5).astype(np.uint8)
            base[iy0:iy1, ix0:ix1] = color

    img = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(img)

    border_px = max(1, pixel_size // 8)   # 边界线宽度

    def diff_color(diff: float) -> tuple[int, int, int]:
        if diff >= 80:
            return (50, 220, 80)    # 绿
        elif diff >= 30:
            return (220, 200, 50)   # 黄
        else:
            return (220, 60, 60)    # 红

    content = {k: v for k, v in cells.items() if not v['is_bg']}

    for (gy, gx), cv in content.items():
        # 右侧相邻格
        if (gy, gx + 1) in content:
            nb = content[(gy, gx + 1)]
            diff = float(np.abs(cv['median'] - nb['median']).sum())
            color = diff_color(diff)
            # 边界 x 位置（两格之间）
            bx = cv['cell_x1']   # 格子右端（含网格线起点）
            y0 = max(0, min(cv['img_y0'], nb['img_y0']))
            y1 = min(h, max(cv['img_y1'], nb['img_y1']))
            draw.line([(bx, y0), (bx, y1)], fill=color, width=border_px)

        # 下方相邻格
        if (gy + 1, gx) in content:
            nb = content[(gy + 1, gx)]
            diff = float(np.abs(cv['median'] - nb['median']).sum())
            color = diff_color(diff)
            by = cv['cell_y1']
            x0 = max(0, min(cv['img_x0'], nb['img_x0']))
            x1 = min(w, max(cv['img_x1'], nb['img_x1']))
            draw.line([(x0, by), (x1, by)], fill=color, width=border_px)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# 8. 综合对比卡
# ─────────────────────────────────────────────────────────────────────────────

def make_report_card(orig:       Image.Image,
                     pp_out:     Image.Image,
                     heatmap:    Image.Image,
                     bdry_viz:   Image.Image,
                     metrics:    dict,
                     name:       str,
                     has_grid:   bool,
                     pixel_size: int,
                     elapsed:    float) -> Image.Image:
    """生成 4 列对比卡（原图 / 像素完美输出 / 纯色热力图 / 边界锐度图）。"""
    THUMB = 360
    INFO_H = 200
    COL_GAP = 10
    HEADER_H = 52
    BG = (20, 20, 26)

    cols = 4
    W = THUMB * cols + COL_GAP * (cols + 1)
    H = HEADER_H + THUMB + INFO_H + 20

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    font_lg = _get_font(18)
    font_sm = _get_font(14)
    font_xs = _get_font(12)

    # ── 标题 ──
    score = metrics.get('score', 0)
    score_color = (80, 220, 80) if score >= 60 else (220, 180, 60) if score >= 30 else (220, 60, 60)
    draw.text((COL_GAP, 10), f"Algo4  |  {name}", fill=(200, 200, 200), font=font_lg)
    draw.text((W - 200, 10),
              f"遵循分: {score:.1f}/100",
              fill=score_color, font=font_lg)

    # ── 4 列缩略图 ──
    labels = ["原图", "像素完美输出", "纯色热力图", "边界锐度图"]
    images = [orig, pp_out, heatmap, bdry_viz]
    for i, (lbl, im) in enumerate(zip(labels, images)):
        x = COL_GAP + i * (THUMB + COL_GAP)
        y = HEADER_H
        thumb = im.convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
        canvas.paste(thumb, (x, y))
        draw.text((x + 4, y + 4), lbl, fill=(240, 240, 200), font=font_sm)

    # ── 指标文字 ──
    INFO_Y = HEADER_H + THUMB + 12
    m = metrics
    cs = metrics.get('center_sample', 4)
    lines = [
        f"pixel_size={pixel_size}  取色={cs}×{cs}中心  "
        f"网格线可见={'是' if has_grid else '否'}  耗时={elapsed:.2f}s",
        f"内容格数: {m.get('n_content',0)} / {m.get('n_total',0)} 总格",
        f"CPR_hard (var<{PURITY_THRESH_HARD}): {m.get('CPR_hard',0)*100:.1f}%   "
        f"CPR_soft (var<{PURITY_THRESH_SOFT}): {m.get('CPR_soft',0)*100:.1f}%",
        f"MCV (格内均方差): {m.get('MCV',0):.1f}   "
        f"BIR (边界/内部比): {m.get('BIR',0):.2f}",
    ]
    legend = [
        "热力图图例: ■深绿=严格纯色  ■黄绿=宽松纯色  ■橙=混浊  ■深红=高度混浊  ■灰=背景",
        "边界图图例: ■绿=边界锐利(遵循好)  ■黄=中等  ■红=边界模糊(遵循差)",
    ]
    for li, txt in enumerate(lines + legend):
        draw.text((COL_GAP, INFO_Y + li * 22), txt,
                  fill=(190, 200, 190) if li < len(lines) else (150, 150, 170),
                  font=font_xs)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 9. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(src_path:        Path,
                 grid_ref_path:   Path  = DEFAULT_GRID_REF,
                 pixel_size_hint: int   = None,
                 out_dir:         Path  = None,
                 save_steps:      bool  = False,
                 center_sample:   int   = 4) -> dict:
    """
    完整的 Algo4 流程。
    返回包含所有评估指标和文件路径的字典。
    """
    t0 = time.time()
    name = src_path.stem
    if out_dir is None:
        out_dir = Path(__file__).parent / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取并去背景 ──────────────────────────────────────────────────────
    img = Image.open(src_path).convert("RGB")
    img_rgba = remove_white_background(img)
    rgb   = np.array(img.convert("RGB"))
    valid = np.array(img_rgba)[:, :, 3] > 0
    h, w  = rgb.shape[:2]
    print(f"\n[Algo4] {name}  size={w}x{h}")

    # ── 2. 确定 pixel_size 和 offset ────────────────────────────────────────
    if pixel_size_hint:
        pixel_size = pixel_size_hint
        offset_x = offset_y = 0
        print(f"  pixel_size=先验 {pixel_size}, offset假设(0,0)")
    else:
        pixel_size, offset_x, offset_y = parse_reference_grid(grid_ref_path)
        print(f"  从参考网格解析: pixel_size={pixel_size}, offset=({offset_x},{offset_y})")

    # ── 3. 检测可见网格线 ────────────────────────────────────────────────────
    has_grid = detect_visible_gridlines(rgb, pixel_size, offset_x, offset_y)
    grid_margin = 1 if has_grid else 0
    print(f"  可见网格线: {'是' if has_grid else '否'}  → grid_margin={grid_margin}")

    # ── 4. 逐格分析 ──────────────────────────────────────────────────────────
    cells = analyze_cells(rgb, valid, pixel_size, offset_x, offset_y,
                          grid_margin, center_sample)
    n_total   = len(cells)
    n_content = sum(1 for c in cells.values() if not c['is_bg'])
    print(f"  格子总数: {n_total}  内容格: {n_content}")

    # ── 5. 计算指标 ──────────────────────────────────────────────────────────
    metrics = compute_metrics(cells)
    metrics['center_sample'] = center_sample
    score = metrics['score']
    print(f"  CPR_hard={metrics['CPR_hard']*100:.1f}%  CPR_soft={metrics['CPR_soft']*100:.1f}%"
          f"  MCV={metrics['MCV']:.1f}  BIR={metrics['BIR']:.2f}  score={score:.1f}/100")

    # ── 6. 像素完美渲染 ──────────────────────────────────────────────────────
    pp_arr = render_pixel_perfect(rgb, valid, cells, has_grid)
    pp_img = Image.fromarray(pp_arr, "RGB")

    # ── 6b. 透明背景版 & 迷你像素版（始终保存）────────────────────────────
    tp_img   = make_transparent(pp_arr, cells, h, w, pixel_size)   # RGBA，透明背景
    mini_img = make_mini_pixel(cells, tp_img, pixel_size)          # 1格=1像素，同步透明

    tp_path   = out_dir / "pixel_perfect_transparent.png"
    mini_path = out_dir / "pixel_mini.png"
    tp_img.save(tp_path)
    mini_img.save(mini_path)
    print(f"  透明版 → {tp_path}  ({tp_img.width}x{tp_img.height})")
    print(f"  迷你版 → {mini_path}  ({mini_img.width}x{mini_img.height})")

    # ── 7. 可视化 ─────────────────────────────────────────────────────────────
    heatmap  = make_purity_heatmap(cells, h, w)
    bdry_viz = make_boundary_viz(cells, h, w, pixel_size, has_grid)

    # ── 8. 步骤图保存 ─────────────────────────────────────────────────────────
    if save_steps:
        img.save(out_dir / "01_original.png")
        pp_img.save(out_dir / "02_pixel_perfect.png")
        heatmap.save(out_dir / "03_purity_heatmap.png")
        bdry_viz.save(out_dir / "04_boundary_sharpness.png")
        print(f"  步骤图保存至 {out_dir}")

    # ── 9. 对比卡 ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    card = make_report_card(img, pp_img, heatmap, bdry_viz,
                            metrics, name, has_grid, pixel_size, elapsed)
    card_path = out_dir / "compare.png"
    card.save(card_path)
    print(f"  对比卡 → {card_path}")

    return {
        'name':          name,
        'score':         score,
        'CPR_hard':      metrics['CPR_hard'],
        'CPR_soft':      metrics['CPR_soft'],
        'MCV':           metrics['MCV'],
        'BIR':           metrics['BIR'],
        'n_content':     metrics['n_content'],
        'n_total':       metrics['n_total'],
        'has_grid':      has_grid,
        'pixel_size':    pixel_size,
        'center_sample': center_sample,
        'elapsed':       elapsed,
        'card_path':     card_path,
        'pp_path':       out_dir / "02_pixel_perfect.png" if save_steps else None,
        'tp_path':       tp_path,
        'mini_path':     mini_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Algo4: 网格遵循度评估")
    ap.add_argument("src", help="输入图片路径")
    ap.add_argument("--grid-ref", default=str(DEFAULT_GRID_REF),
                    help="参考网格图路径（默认 pixel_grid/grid_1px_black.png）")
    ap.add_argument("--pixel-size", type=int, default=None,
                    help="强制指定 pixel_size（跳过自动检测）")
    ap.add_argument("--out-dir", default=None, help="输出目录")
    ap.add_argument("--save-steps", action="store_true", help="保存中间步骤图")
    ap.add_argument("--center-sample", type=int, default=4,
                    help="中心取色窗口大小（默认4，即4×4）")
    args = ap.parse_args()

    result = run_pipeline(
        src_path        = Path(args.src),
        grid_ref_path   = Path(args.grid_ref),
        pixel_size_hint = args.pixel_size,
        out_dir         = Path(args.out_dir) if args.out_dir else None,
        save_steps      = args.save_steps,
        center_sample   = args.center_sample,
    )
    print(f"\n综合遵循分: {result['score']:.1f}/100")


if __name__ == "__main__":
    main()
