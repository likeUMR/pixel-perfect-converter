"""
shared_utils.py — 三种像素艺术检测算法的公共工具库
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import deque, Counter


# ──────────────────────────────────────────────────────────────────────────────
# 背景去除
# ──────────────────────────────────────────────────────────────────────────────

def remove_white_background(img: Image.Image, threshold: int = 235) -> Image.Image:
    """BFS 从四边泛洪填充，去除白色背景 → 透明。"""
    img_rgba = img.convert("RGBA")
    data = np.array(img_rgba, dtype=np.uint8)
    h, w = data.shape[:2]
    r = data[:, :, 0].astype(np.int32)
    g = data[:, :, 1].astype(np.int32)
    b = data[:, :, 2].astype(np.int32)
    is_white = (r >= threshold) & (g >= threshold) & (b >= threshold)
    background = np.zeros((h, w), dtype=bool)
    queue = deque()

    def enq(y, x):
        if 0 <= y < h and 0 <= x < w and not background[y, x] and is_white[y, x]:
            background[y, x] = True
            queue.append((y, x))

    for x in range(w):
        enq(0, x); enq(h - 1, x)
    for y in range(h):
        enq(y, 0); enq(y, w - 1)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in dirs:
            enq(cy + dy, cx + dx)

    result = data.copy()
    result[background, 3] = 0
    return Image.fromarray(result, "RGBA")


# ──────────────────────────────────────────────────────────────────────────────
# 内容边界框
# ──────────────────────────────────────────────────────────────────────────────

def get_content_bbox(valid: np.ndarray, padding: int = 8):
    """返回非透明像素的 bounding box (x0, y0, x1, y1)。"""
    ys, xs = np.where(valid)
    if len(ys) == 0:
        h, w = valid.shape
        return 0, 0, w, h
    h, w = valid.shape
    return (max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(w, int(xs.max()) + padding + 1),
            min(h, int(ys.max()) + padding + 1))


# ──────────────────────────────────────────────────────────────────────────────
# 像素尺寸检测（跳变间距和声分析 + 梯度偏移精修）
# ──────────────────────────────────────────────────────────────────────────────

def detect_pixel_size(img_rgba: Image.Image,
                      max_size: int = 32,
                      verbose: bool = False) -> tuple[int, int, int]:
    """
    返回 (pixel_size, offset_x, offset_y)。
    只在内容区域内分析，速度较快。
    """
    data = np.array(img_rgba)
    valid = data[:, :, 3] > 0 if data.shape[2] == 4 else np.ones(data.shape[:2], dtype=bool)
    rgb = data[:, :, :3].astype(np.float32)
    x0c, y0c, x1c, y1c = get_content_bbox(valid, padding=2)
    rgb_c = rgb[y0c:y1c, x0c:x1c]
    valid_c = valid[y0c:y1c, x0c:x1c]
    h, w = rgb_c.shape[:2]
    DIFF_THRESH = 35.0

    # 和声分数
    gap_counter: Counter = Counter()
    for y in range(0, h, max(1, h // 100)):
        row = rgb_c[y].astype(float)
        vrow = valid_c[y]
        prev = None
        for x in range(1, w):
            if not (vrow[x] and vrow[x - 1]):
                continue
            d = float(np.sum(np.abs(row[x] - row[x - 1])))
            if d > DIFF_THRESH:
                if prev is not None:
                    gap = x - prev
                    if 2 <= gap <= max_size * 3:
                        gap_counter[gap] += 1
                prev = x
    for x in range(0, w, max(1, w // 100)):
        col = rgb_c[:, x].astype(float)
        vcol = valid_c[:, x]
        prev = None
        for y in range(1, h):
            if not (vcol[y] and vcol[y - 1]):
                continue
            d = float(np.sum(np.abs(col[y] - col[y - 1])))
            if d > DIFF_THRESH:
                if prev is not None:
                    gap = y - prev
                    if 2 <= gap <= max_size * 3:
                        gap_counter[gap] += 1
                prev = y

    scores = {sz: sum(gap_counter[sz * k] for k in range(1, max_size * 3 // sz + 1))
              for sz in range(2, max_size + 1)}
    best_size = max(scores, key=scores.get)

    if verbose:
        top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
        print(f"  Top-5 pixel sizes: {top5}")

    # 梯度偏移精修
    rgb_f = rgb.astype(np.float32)
    h2, w2 = rgb_f.shape[:2]
    best_s_score, best_ox, best_oy = float("inf"), 0, 0
    for oy in range(best_size):
        for ox in range(best_size):
            hd = np.sum(np.abs(rgb_f[:, 1:] - rgb_f[:, :-1]), axis=2)
            hv = valid[:, 1:] & valid[:, :-1]
            xs2 = np.arange(w2 - 1)
            hb = ((xs2 - ox + 1) % best_size == 0)[np.newaxis, :]
            hi_v = hd[hv & ~hb]
            hb_v = hd[hv & hb]
            vd = np.sum(np.abs(rgb_f[1:] - rgb_f[:-1]), axis=2)
            vv = valid[1:] & valid[:-1]
            ys2 = np.arange(h2 - 1)
            vb = ((ys2 - oy + 1) % best_size == 0)[:, np.newaxis]
            vi_v = vd[vv & ~vb]
            vb_v = vd[vv & vb]
            interior = np.concatenate([hi_v, vi_v])
            boundary = np.concatenate([hb_v, vb_v])
            if len(interior) == 0 or len(boundary) == 0:
                continue
            s = float(np.mean(interior)) / (float(np.mean(boundary)) + 1e-6)
            if s < best_s_score:
                best_s_score, best_ox, best_oy = s, ox, oy

    return best_size, best_ox, best_oy


# ──────────────────────────────────────────────────────────────────────────────
# 步骤图保存（带底部说明栏）
# ──────────────────────────────────────────────────────────────────────────────

def _get_font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def save_step(img: Image.Image, out_dir: Path, name: str, label: str = "") -> Path:
    """保存中间步骤图，底部附加说明文字栏。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    if label:
        bar_h = 40
        canvas = Image.new("RGBA", (img.width, img.height + bar_h), (22, 22, 28, 255))
        canvas.paste(img.convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((10, img.height + 8), label, fill=(220, 220, 200, 255), font=_get_font(15))
        canvas.save(path)
    else:
        img.convert("RGBA").save(path)
    print(f"  -> {name}.png  [{label[:60]}]" if label else f"  -> {name}.png")
    return path


def make_base_canvas(img_rgba: Image.Image, bg: tuple = (38, 38, 42)) -> Image.Image:
    """在深色背景上合成带透明度的图像。"""
    base = Image.new("RGBA", img_rgba.size, bg + (255,))
    if img_rgba.mode == "RGBA":
        base.paste(img_rgba, mask=img_rgba.split()[3])
    else:
        base.paste(img_rgba)
    return base


def draw_grid(draw: ImageDraw.ImageDraw, w: int, h: int,
              offset_x: int, offset_y: int, pixel_size: int,
              color=(255, 255, 255, 50), line_w: int = 1):
    """在 draw 对象上绘制网格线。"""
    for x in range(offset_x, w, pixel_size):
        draw.line([(x, 0), (x, h - 1)], fill=color, width=line_w)
    for y in range(offset_y, h, pixel_size):
        draw.line([(0, y), (w - 1, y)], fill=color, width=line_w)
