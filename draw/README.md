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

## 论文 4.3 节候选图

新增脚本：

- `draw/paper_figures_4_3/generate_figures_4_3.py`
  - 面向论文 4.3 消融实验生成候选图。
  - 同时使用聚合消融结果、窗口级散点数据、结构指标逐记录数据和代表性预测轨迹。
  - 默认只使用论文当前 4.3 节涉及的五个核心模型：`chronos2`、`sundial`、`timesfm2p0`、`timesfm2p5`、`visiontspp`。

在仓库根目录运行：

```bash
python draw/paper_figures_4_3/generate_figures_4_3.py
```

如需在窗口级图中包含已有 `Kairos` 结果：

```bash
python draw/paper_figures_4_3/generate_figures_4_3.py --include_extra_models
```

主要输入数据：

- `results_analysis/ablation/ablation_*.csv`
- `draw/outputs_by_dataset/*/*.csv`
- `results_analysis/structure_metric_ablation/structure_metric_ablation_records.csv`
- `results_analysis/structure_metric_ablation/structure_metric_ablation_summary.csv`
- `tools/Sample/sample_forward_linear_kalman_forecast_*.csv`

输出根目录：

- `draw/paper_figures_4_3/figure_manifest.csv`：本次生成的图片清单。
- `draw/paper_figures_4_3/data/`：为绘图保留的派生数据，便于复查图中点和统计量。
- `draw/paper_figures_4_3/aggregate_overview/`：聚合消融趋势图，用于定位主要结论。
- `draw/paper_figures_4_3/window_landscape/`：窗口级散点图，每个点对应一个预测窗口。
- `draw/paper_figures_4_3/error_distributions/`：窗口级误差分布图，展示方法间离散性和长尾。
- `draw/paper_figures_4_3/forecast_cases/`：代表性预测轨迹图，展示同一窗口下不同填补输入对预测曲线的影响。
- `draw/paper_figures_4_3/structure_conditions/`：结构指标条件图，用于解释填补影响与序列结构差异之间的关系。

当前生成的候选图：

- `aggregate_overview/fig4_3_1_missing_ratio_sensitivity.{png,svg}`
- `aggregate_overview/fig4_3_2_method_gap_by_difficulty.{png,svg}`
- `window_landscape/fig4_3_3_window_error_landscape.{png,svg}`
- `error_distributions/fig4_3_4_window_error_distribution.{png,svg}`
- `forecast_cases/fig4_3_5_representative_forecast_case.{png,svg}`
- `structure_conditions/fig4_3_6_structure_condition_gain.{png,svg}`
- `structure_conditions/fig4_3_7_structure_channel_heatmap.{png,svg}`
