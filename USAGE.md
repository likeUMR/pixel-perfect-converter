# 使用说明

## 概述

本工具的完整工作流分为两个阶段：

1. **生成阶段**：用 AI 大模型（如 nanobanana2）配合参考网格图生成像素风格图片
2. **处理阶段**：用 `algo4` 评估网格遵循度，生成 pixel perfect 修复版本

---

## 第一阶段：用 AI 生成像素图

### 准备参考网格图

参考网格图位于：

```
pixel_grid/grid_1px_black.png
```

这是一张 512×512 的图，带有 1px 黑色网格线，每 8px 一格，定义了逻辑像素的大小和位置。

### 在 nanobanana2 中生图

在 nanobanana2（或任何支持图片参考/ControlNet 的 AI 工具）中：

1. **上传参考图**：将 `grid_1px_black.png` 作为参考图或 ControlNet 输入
2. **写 Prompt**：

```
生成一个白色背景像素风格的[描述内容]（参考图片对像素进行填充，保留网格）
```

示例：

```
生成一个白色背景像素风格的小猫咪（参考图片对像素进行填充，保留网格）
```

```
生成一个白色背景像素风格的骑士角色，正面站立（参考图片对像素进行填充，保留网格）
```

**关键点说明：**

- "白色背景"：让背景保持纯白，便于后处理去除
- "像素风格"：触发 AI 的像素艺术风格
- "参考图片对像素进行填充"：让 AI 理解参考图的网格是像素边界，每格填一个颜色
- "保留网格"：让 AI 在输出中保留网格线（可见的网格线有助于后续算法准确识别像素边界）

### 生成质量判断

生成后，肉眼观察图片：
- **好的情况**：能看到清晰的 8×8 像素格，每格颜色均匀，网格线明显
- **差的情况**：像素格模糊，颜色渐变严重，或完全没有网格线

可以用本工具的评分来辅助判断（见下文）。

---

## 第二阶段：评估和修复生成的图片

### 安装依赖

```bash
pip install Pillow numpy scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Python 3.10+ 必须。

### 处理单张图片

```bash
python algo4/run.py <图片路径> --grid-ref pixel_grid/grid_1px_black.png
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--grid-ref` | 必填 | 参考网格图路径 |
| `--pixel-size` | 从参考图自动检测 | 锁定逻辑像素大小（推荐传 8） |
| `--out-dir` | `algo4/results/<图片名>` | 结果输出目录 |
| `--center-sample` | 4 | 取色时采样的中心区域大小（像素） |
| `--save-steps` | 否 | 是否保存中间步骤图片 |

示例：

```bash
python algo4/run.py pictures/my_art.png --grid-ref pixel_grid/grid_1px_black.png --pixel-size 8
```

### 批量处理整个目录

```bash
python algo4/evaluate.py --src pictures/ --grid-ref pixel_grid/grid_1px_black.png --pixel-size 8
```

批量处理会额外生成：
- `algo4/results/all_evaluate.png`：所有图片的对比卡片纵向拼接
- `algo4/results/summary.txt`：所有图片的评分汇总

---

## 输出文件说明

每张图片的结果保存在 `algo4/results/<图片名>/` 目录下：

| 文件名 | 说明 |
|--------|------|
| `01_original.png` | 原始图片（去白底后） |
| `02_pixel_perfect.png` | 像素完美修复版（白色背景，8×8 纯色块） |
| `pixel_perfect_transparent.png` | 像素完美修复版（透明背景 RGBA PNG） |
| `pixel_mini.png` | 迷你像素版（每个逻辑像素 = 1 图像像素，透明背景） |
| `03_purity_heatmap.png` | 纯色热力图（显示每个格子的颜色纯度） |
| `04_boundary_viz.png` | 边界锐利度可视化（显示格子间颜色差异） |
| `report_card.png` | 综合对比卡片（汇总以上所有结果 + 评分） |

---

## 评分指标说明

| 指标 | 含义 | 好的范围 |
|------|------|---------|
| `CPR_hard` | 严格纯色率：格内方差 < 200 的格子比例 | > 80% |
| `CPR_soft` | 宽松纯色率：格内方差 < 2000 的格子比例 | > 90% |
| `MCV` | 平均格内方差（越低越纯色） | < 500 |
| `BIR` | 边界/内部比：格间颜色差 vs 格内颜色差（越高边界越锐利） | > 4 |
| `score` | 综合评分 = CPR_soft×40 + min(BIR/6,1)×60 | > 70 |

**综合评分解读：**
- 90+ 分：接近 pixel perfect，AI 非常好地遵循了网格
- 70–90 分：总体不错，局部有一些模糊
- 50–70 分：一般，需要较多修复
- < 50 分：较差，AI 基本没有遵循网格

---

## 典型工作流示例

```
1. 用 nanobanana2 生成：白色背景像素风格的骑士（参考 grid_1px_black.png，保留网格）

2. 保存生成图到 pictures/knight.png

3. 批量评估：
   python algo4/evaluate.py --src pictures/ --grid-ref pixel_grid/grid_1px_black.png --pixel-size 8

4. 查看 algo4/results/knight/report_card.png → 分数 85 分，不错

5. 使用结果：
   - pixel_perfect_transparent.png → 透明背景版，可直接用于游戏/合成
   - pixel_mini.png               → 迷你版，可放大用于游戏 sprite
```

---

## 常见问题

**Q：为什么 pixel_mini.png 部分区域是透明的？**

A：背景（白色区域）会被自动识别并设为透明。如果角色边缘有问题，可以调整 `BG_THRESH` 参数（在 `algo4/run.py` 顶部）。

**Q：score 很低但图片看起来还行？**

A：可能是 AI 没有保留网格线，导致算法检测不到边界锐利度。可以在 Prompt 中强调"保留网格"，或检查 `report_card.png` 中的 boundary 可视化。

**Q：如何处理非 512×512 的图？**

A：直接传入即可，算法会根据参考网格图自动推断像素大小。建议始终传入 `--pixel-size 8` 以锁定像素大小，避免检测误差。
