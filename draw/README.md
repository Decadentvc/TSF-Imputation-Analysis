# `draw` 可视化脚本说明

## 脚本划分

- `draw/visualized_results_by_mode.py`
  - 一次选择一种 `prediction_mode`
  - 一张图里包含该模式下的所有数据集
  - 点颜色表示数据集
- `draw/visualized_results_by_dataset.py`
  - 一次可包含多种 `prediction_mode`（填补方法）
  - 每个数据集单独一张图
  - 点颜色表示填补方法，且同一填补方法在所有图中颜色一致

## 坐标定义（两脚本一致）

- 横轴：history 与 prediction 的分布差异（6 个 STL 指标标准化后做 RMS 欧氏聚合）
- 纵轴：
  - `by_mode`：`clean` 时为 `sMAPE[0.5]`，其他模式为 `sMAPE[0.5](mode)-sMAPE[0.5](clean)`
  - `by_dataset`：统一使用 `sMAPE[0.5](mode)-sMAPE[0.5](clean)`（若包含 `clean`，其值按 0 处理）

## 使用方式

在仓库根目录运行。

按填补方法分图（旧逻辑，已改名）：

```bash
python draw/visualized_results_by_mode.py --model chronos2 --prediction_mode backward
```

按数据集分图（新逻辑）：

```bash
python draw/visualized_results_by_dataset.py --model chronos2
```

可选参数示例：

```bash
python draw/visualized_results_by_dataset.py --model chronos2 \
  --modes zero,mean,forward,backward,linear \
  --datasets ETTh1,ETTh2
```

## 输出目录

两个脚本默认输出到不同目录：

- `by_mode`
  - 图片：`draw/outputs_by_model/<model>/<prediction_mode>_window_gap_scatter.png`
  - 点数据：`draw/outputs_by_model/<model>/<prediction_mode>_window_gap_scatter.csv`
- `by_dataset`
  - 图片：`draw/outputs_by_dataset/<model>/<dataset>_window_gap_scatter_by_imputation.png`
  - 点数据：`draw/outputs_by_dataset/<model>/<dataset>_window_gap_scatter_by_imputation.csv`
