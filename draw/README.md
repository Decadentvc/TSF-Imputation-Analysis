# `draw/visualized_results.py` 说明

## 功能
该脚本用于做窗口级可视化分析，绘制：

- 横轴：回顾窗口（history）与预测窗口（prediction）的分布差异
- 纵轴：预测误差（`sMAPE[0.5]`）或相对 clean 的误差差异
- 点：每个点对应一个滑动窗口
- 颜色：不同数据集使用不同颜色

## 数据来源
- `results_analysis/<model>/{history,prediction}/*.csv`
- `data/Intermediate_Predictions/<model>/...`
- `data/datasets/ori/<dataset>.csv`
- `data/datasets/dataset_properties.json`

## 横轴计算（新）
为了平等看待 6 个 STL 指标（数量级不同），脚本改为：

1. 对每个指标，使用当前组合下 history/prediction 窗口池化值计算标准差 `std`
2. 对窗口差值做标准化：`(history - prediction) / std`
3. 将 6 个标准化差值做欧氏聚合（RMS 形式）：
   `distribution_gap = sqrt(mean(delta_norm^2))`

这个做法与 TIME 文中“跨指标可比、先归一化再聚合”的思想一致，避免某一指标因量纲过大主导结果。

## 纵轴计算（新）
- `prediction_mode=clean`：纵轴为 `sMAPE[0.5]`
- `prediction_mode!=clean`：纵轴为 `sMAPE[0.5](mode) - sMAPE[0.5](clean)`

sMAPE 计算为窗口内逐点平均：
`100 * mean( 2*|y_pred - y_true| / (|y_pred| + |y_true|) )`

## 使用方式
在仓库根目录运行：

```bash
python draw/visualized_results.py --model chronos2 --prediction_mode backward
```

常用参数：
- `--model`：如 `chronos2` / `sundial` / `timesfm2p5`
- `--prediction_mode`：如 `clean` / `mean` / `forward` / `backward` / `linear`
- `--max_points`：可选，绘图采样点上限
- `--output`：可选，自定义输出图片路径

## 输出目录（新）
现在按模型分类输出：

- 图片：`draw/outputs/<model>/<prediction_mode>_window_gap_scatter.png`
- 点数据：`draw/outputs/<model>/<prediction_mode>_window_gap_scatter.csv`

例如：
- `draw/outputs/chronos2/backward_window_gap_scatter.png`
- `draw/outputs/chronos2/backward_window_gap_scatter.csv`
