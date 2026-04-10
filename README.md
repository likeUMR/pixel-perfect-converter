# Pixel Perfect Checker

A toolkit for analyzing and repairing AI-generated pixel art images.

AI image generators (Stable Diffusion, Midjourney, etc.) can produce convincing pixel art, but the output is never truly "pixel-perfect" — each logical pixel is rendered with sub-pixel anti-aliasing, slight color variation, and grid drift across the canvas. This project detects the logical pixel grid, measures purity, and reconstructs a clean pixel-perfect version.

---

## The Problem

| Original AI output | What we want |
|---|---|
| Each logical pixel is ≈ N×N screen pixels, but with anti-aliasing at edges, color noise inside, and the grid position drifting slightly across the image | Every logical pixel is exactly N×N screen pixels of a single flat color, perfectly aligned to a global grid |

---

## Repository Structure

```
pixel_perfect_checker.py   # Original all-in-one checker (white-bg removal, grid detect, repair)
shared_utils.py            # Common utilities shared by all algorithms

algo1/run.py               # Algorithm 1: Bean Seed (撒豆算法)
algo2/run.py               # Algorithm 2: Block Grid  (分块网格对齐)
algo3/run.py               # Algorithm 3: Gradient + Hough grid fitting
```

A sample input image (`20260410013055_8bbbc8af.png`) is included for testing.

---

## Algorithms

### `pixel_perfect_checker.py` — Baseline Checker

The original single-file tool. Handles the full pipeline:

1. **White background removal** — BFS flood-fill from image edges, converts white pixels to transparent
2. **Pixel size detection** — Transition-gap harmony analysis: scans color transitions, computes which period N has the most "harmonious" gap distribution (i.e., gaps are multiples of N)
3. **Grid offset refinement** — Gradient ratio method: finds (offset_x, offset_y) that minimizes interior/boundary gradient ratio
4. **Purity check** — Measures color variance within each N×N grid cell
5. **Pixel-perfect output** — Snaps each cell to its median color; also exports a native-resolution sprite

```bash
python pixel_perfect_checker.py <image> [--pixel-size N] [--purity-threshold N]
```

---

### Algorithm 1: Bean Seed (`algo1/run.py`) ⭐ Recommended

**Core idea:** Instead of assuming a fixed global grid, find each logical pixel *independently* using dense seed points, then align them collectively.

**Pipeline (10 steps, each saved as an image):**

1. Remove white background
2. Cast dense seed points (stride ≈ 3px) across the content area
3. For each seed, find the **largest pure-color square** it belongs to — using precomputed variance maps at every scale
4. Merge overlapping/duplicate squares (keep lowest-variance representative)
5. **Infer pixel size** from the merged square size distribution (IQR-based outlier removal, take median)
6. Normalize all inlier squares to exactly `pixel_size × pixel_size`
7. Split oversized outlier squares into `k × k` sub-pixels
8. **Grid alignment** — find global offset via histogram-mode of `(center_x mod pixel_size)`
9. Fill empty neighbor cells (use actual image color, fall back to random neighbor)
10. Final pixel-perfect output

**Key advantage:** Robust to local grid drift. Since each pixel is found independently, areas where the grid shifts slightly still get correct local colors. The grid alignment step then finds the best global grid that fits all detected pixels.

```bash
python algo1/run.py <image> [--pixel-size N] [--stride N] [--var-thresh N]
```

---

### Algorithm 2: Block Grid (`algo2/run.py`)

**Core idea:** Try multiple ways of partitioning the image into blocks (1×1, 2×2, 3×3, 3×4, 4×4, …), run independent grid detection in each block, and pick the partition that gives the most consistent pixel sizes with the best gradient alignment scores.

**Key output:** Each block reports its local `(pixel_size, offset_x, offset_y)`. Differences in offset between blocks directly show the **grid drift** across the image.

```bash
python algo2/run.py <image> [--pixel-size N]
```

---

### Algorithm 3: Gradient + Hough (`algo3/run.py`)

**Core idea:** Treat pixel boundaries as a periodic signal. Compute the Sobel gradient, project onto horizontal/vertical axes, then use FFT to find the dominant period (= pixel size) and cross-correlation to find the phase (= grid offset).

**Local drift analysis:** Slide a window across the image, detect the local phase in each band, and compute the deviation from the global phase. This produces a **drift field** showing exactly how many pixels the grid shifts in each region.

**Adaptive rendering:** When generating the final output, compensate for local drift by shifting the sampling window according to the measured drift at each grid cell location.

```bash
python algo3/run.py <image> [--pixel-size N]
```

---

## Installation

```bash
pip install Pillow numpy scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Python 3.10+ required (uses `X | Y` union type hints).

---

## Example Results

Running all three algorithms on the sample image (`pixel_size=11`, 768×1376 input):

| Algorithm | Pixel size detected | Notes |
|---|---|---|
| Baseline checker | 11px | Global grid, purity 14% (AI anti-aliasing expected) |
| **Bean Seed** | 11px | 744 unique pixels found, 858 grid cells in final output |
| Block Grid | 11px (all blocks) | Best partition: 3×4; reveals per-block offset drift |
| Gradient Hough | 11px | Drift: X ∈ [−5, +5]px, Y ∈ [−3, +4]px across image |

---

## Known Limitations

- **Same-color adjacent pixels** cannot be separated without prior knowledge of the original pixel art — any method will merge them
- **AI anti-aliasing** at pixel boundaries reduces purity; this is expected and not a bug
- **Grid drift** detected by Algorithm 2/3 is real but small (< half a pixel-size), so a global grid still gives acceptable results for most images
- Works best with white-background pixel art; colored backgrounds need manual threshold adjustment

---

## License

MIT
