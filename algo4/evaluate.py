#!/usr/bin/env python3
"""
algo4/evaluate.py — 批量评估 AI 绘图的网格遵循程度
====================================================

对指定目录（递归）内所有图片运行 Algo4，生成：
  - 每张图片的对比卡（results/<name>/compare.png）
  - 所有图片合并的大对比图（results/all_evaluate.png）
  - 文字摘要报告（results/summary.txt）

用法：
  cd algo4
  python evaluate.py D:\\Project\\tools\\pictures [--pixel-size 8]
  python evaluate.py D:\\Project\\tools\\pictures --save-steps
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import _get_font
from run import run_pipeline, DEFAULT_GRID_REF

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# 合并所有对比卡为一张大图
# ─────────────────────────────────────────────────────────────────────────────

def combine_cards(card_paths: list, out_path: Path):
    """垂直堆叠所有对比卡，在顶部加汇总标题行。"""
    cards = [Image.open(p) for p in card_paths]
    if not cards:
        return

    w = max(c.width for c in cards)
    total_h = sum(c.height for c in cards) + 20 * len(cards)

    BG = (15, 15, 20)
    canvas = Image.new("RGB", (w, total_h), BG)
    y = 0
    for card in cards:
        if card.width < w:
            bg = Image.new("RGB", (w, card.height), BG)
            bg.paste(card, (0, 0))
            card = bg
        canvas.paste(card, (0, y))
        y += card.height + 20

    canvas.save(out_path)
    print(f"\n合并对比图 → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 摘要报告
# ─────────────────────────────────────────────────────────────────────────────

def make_summary(results: list, out_path: Path, pixel_size: int):
    lines = [
        "=" * 72,
        f"Algo4 网格遵循度评估摘要  (pixel_size={pixel_size})",
        "=" * 72,
        f"{'图片名':<35} {'分数':>6} {'CPR_s':>6} {'MCV':>8} {'BIR':>5} {'有网格':>5} {'耗时':>6}",
        "-" * 72,
    ]
    for r in sorted(results, key=lambda x: -x['score']):
        lines.append(
            f"{r['name']:<35} {r['score']:>6.1f} "
            f"{r['CPR_soft']*100:>5.1f}% "
            f"{r['MCV']:>8.1f} "
            f"{r['BIR']:>5.2f} "
            f"{'Y' if r['has_grid'] else 'N':>5} "
            f"{r['elapsed']:>5.2f}s"
        )
    if results:
        scores = [r['score'] for r in results]
        lines += [
            "-" * 72,
            f"{'平均':>35} {np.mean(scores):>6.1f}  "
            f"最高={max(scores):.1f}  最低={min(scores):.1f}",
        ]
    lines += [
        "=" * 72,
        "",
        "分数说明:",
        "  score = CPR_soft*40 + min(BIR/6, 1)*60",
        "  CPR_soft : 格内方差<2000 的内容格占比（宽松纯色率）",
        "  MCV      : 格内平均方差（越低越纯，单位：像素通道方差²）",
        "  BIR      : 格间色差均值 / 格内标准差均值（越高越好，理想>4）",
        "  有网格   : AI 是否在输出图像中保留了参考网格线",
    ]
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    # 打印到终端时转义无法编码的字符
    print(text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))
    print(f"\n摘要报告 -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Algo4 批量网格遵循度评估")
    ap.add_argument("img_dir", help="图片目录（递归扫描）")
    ap.add_argument("--grid-ref", default=str(DEFAULT_GRID_REF),
                    help="参考网格图路径")
    ap.add_argument("--pixel-size", type=int, default=8,
                    help="pixel_size（默认 8）")
    ap.add_argument("--out-dir", default=None,
                    help="评估结果目录（默认 algo4/results/）")
    ap.add_argument("--save-steps", action="store_true",
                    help="保存每张图的中间步骤图")
    ap.add_argument("--center-sample", type=int, default=4,
                    help="中心取色窗口大小（默认4，即4×4）")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图片（递归）
    paths = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in SUPPORTED])
    if not paths:
        print(f"未找到支持的图片文件：{img_dir}")
        return

    print(f"找到 {len(paths)} 张图片，开始评估…\n")

    results = []
    card_paths = []

    for src in paths:
        img_out = out_dir / src.stem
        try:
            r = run_pipeline(
                src_path        = src,
                grid_ref_path   = Path(args.grid_ref),
                pixel_size_hint = args.pixel_size,
                out_dir         = img_out,
                save_steps      = args.save_steps,
                center_sample   = args.center_sample,
            )
            results.append(r)
            card_paths.append(r['card_path'])
        except Exception as e:
            print(f"  [ERROR] {src.name}: {e}")

    # 合并 & 报告
    if card_paths:
        combine_cards(card_paths, out_dir / "all_evaluate.png")
    make_summary(results, out_dir / "summary.txt", args.pixel_size)


if __name__ == "__main__":
    main()
