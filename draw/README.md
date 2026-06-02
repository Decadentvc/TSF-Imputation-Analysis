# draw 图表目录说明

本目录保存论文图、早期 4.2 图组、窗口级诊断图、绘图脚本和派生数据。当前建议优先使用 `paper_figures_4_2/` 与 `paper_figures_4_3/` 中的论文候选图；`section_4_2_original/` 保存从原 `figures/` 迁入的早期 4.2 图组，适合对照或作为备用材料；`outputs_by_model/` 与 `outputs_by_dataset/` 是批量窗口级诊断图。

## 目录结构

`paper_figures_4_2/` 面向论文 4.2 节，包含 7 张候选图和对应派生 CSV。`paper_figures_4_3/` 面向论文 4.3 节，包含 7 张候选图和对应派生 CSV。`section_4_2_original/` 保存原 `figures/` 目录迁入的早期 4.2 图、汇总 CSV 和数据清单。`outputs_by_model/` 按模型和填补方法组织窗口级散点图。`outputs_by_dataset/` 按模型和数据集组织窗口级散点图。`visualized_results_by_mode.py` 与 `visualized_results_by_dataset.py` 是早期批量诊断图脚本。

PNG 和 SVG 表示同一图的两种格式。用于 Word 文档时通常选 PNG；需要进一步编辑、排版或投稿矢量图时选 SVG。

## 4.2 节当前建议图序

### 图 4.2-1 模型稳健性分布

文件：`paper_figures_4_2/model_robustness/fig4_2_1_model_robustness_distribution.{png,svg}`。

该图按模型汇总所有填补情形相对 clean 输入的预测 MSE 变化。横轴是 `MSE change vs clean (%)`，0 左侧表示填补后预测误差低于 clean，右侧表示误差升高。箱线图表示主体分布，红三角表示第 90 分位风险。VisionTS++ 的中位 MSE 变化约为 -3.17%，整体最稳；TimesFM-2.0 的中位变化约为 13.97%，第 90 分位约为 122.75%，右尾风险最高。

### 图 4.2-2 点级填补误差与下游预测误差

文件：`paper_figures_4_2/reconstruction_vs_forecast/fig4_2_2_reconstruction_vs_forecast_scatter.{png,svg}`。

该图展示点级填补 sMAPE 与下游预测 MSE 变化之间的关系。横轴是点级填补 sMAPE，纵轴是相对 clean 的 MSE 变化，颜色和形状区分填补方法，黑线为分箱中位数。图中趋势显示填补误差升高时预测风险总体上升，同时点云离散度很高，说明单点重构质量只能提供部分解释。

### 图 4.2-3 模型与填补方法偏好

文件：`paper_figures_4_2/method_preference/fig4_2_3_model_method_preference.{png,svg}`。

左侧热力图表示每个模型在每种填补方法下的中位 MSE 变化，蓝色表示中位误差低于 clean，红色表示中位误差高于 clean；每行加粗数字是该模型下中位表现最好的方法。右侧柱图统计在“模型、数据集、缺失率”组合中各方法成为最佳方法的次数。SAITS 最多，为 117 次；Mean 为 86 次，GP-RBF 为 70 次，Kalman-ARIMA 为 68 次。

### 图 4.2-4 缺失率轨迹

文件：`paper_figures_4_2/missing_ratio/fig4_2_4_missing_ratio_trajectories.{png,svg}`。

每条细线是一组“模型、数据集、方法”在 10%、20%、30% 缺失率下的 MSE 变化轨迹。红色表示严格递增，灰色表示非单调，蓝色表示严格递减，黑线表示总体中位数。共有 508 条严格递增、666 条非单调、170 条严格递减轨迹。该图说明缺失率升高会放大总体风险，同时局部组合的变化形态存在明显差异。

### 图 4.2-5 数据集敏感性排序

文件：`paper_figures_4_2/dataset_sensitivity/fig4_2_5_dataset_sensitivity_ranked.{png,svg}`。

该图按数据集汇总相对 clean 的 MSE 变化，并按中位数排序。点表示中位数，横线表示四分位范围。traffic、Finland_Traffic_15T、current_velocity_5T 对填补最敏感；azure2019_U_5T、Port_Activity_D、Coastal_T_S_15T 的中位变化为负或接近 0。

### 图 4.2-6 结构扰动与重构误差解释力

文件：`paper_figures_4_2/structure_explanation/fig4_2_6_structure_vs_reconstruction_explanation.{png,svg}`。

该图并列比较两种解释变量与预测误差变化的关系。左图横轴是点级填补 sMAPE，右图横轴是历史序列结构漂移，纵轴均为相对 clean 的 MSE 变化。结构漂移的组内 Spearman 中位数为 0.381，正相关组占 73.8%；点级填补 sMAPE 分别为 0.357 和 70.8%。图中证据支持将结构变化作为解释填补影响的重要变量。

### 图 4.2-7 预测输出结构偏移

文件：`paper_figures_4_2/prediction_structure/fig4_2_7_prediction_structure_shift.{png,svg}`。

该图展示预测输出相对填补后历史输入的结构指标偏移。左侧包括趋势强度、趋势线性、季节强度、季节相关、残差自相关的中位偏移和四分位范围，右侧单独展示谱熵。季节强度、季节相关、趋势强度多数为正偏移，谱熵中位偏移约为 -9301，说明预测输出在若干结构指标上存在稳定偏移。

## 4.3 节当前建议图序

### 图 4.3-1 缺失率消融总览

文件：`paper_figures_4_3/aggregate_overview/fig4_3_1_missing_ratio_sensitivity.{png,svg}`。

该图汇总缺失率消融结果。左图按模型画缺失率与平均 NRMSE 的关系，右图按填补方法画缺失率与平均 weighted quantile loss 的关系。左图显示 Sundial 在 60% 缺失率处明显升高，VisionTS++ 整体 NRMSE 较高；右图显示 SAITS 在多数缺失率下损失较低，GP-RBF 在 60% 处明显恶化。

### 图 4.3-2 受控难度因素下的方法差距

文件：`paper_figures_4_3/aggregate_overview/fig4_3_2_method_gap_by_difficulty.{png,svg}`。

该图展示不同难度因素下填补方法选择造成的 NRMSE 差距。纵轴是同一模型、数据集、难度条件内，不同填补方法 NRMSE 的最大值与最小值之差再取平均。缺失率升到 60% 时差距最大；预测步长和上下文长度增加时，方法间差距也上升。

### 图 4.3-3 窗口级误差地形

文件：`paper_figures_4_3/window_landscape/fig4_3_3_window_error_landscape.{png,svg}`。

每个点表示一个预测窗口，横轴是历史与预测分布差异，纵轴是相对 clean 的 sMAPE 差值，分面对应模型，颜色和形状对应 Mean、Forward、Linear、Backward。该图用于观察同一模型内不同窗口的误差离散性和长尾风险。

### 图 4.3-4 窗口级误差分布

文件：`paper_figures_4_3/error_distributions/fig4_3_4_window_error_distribution.{png,svg}`。

该图按填补方法汇总窗口级 sMAPE 差值分布。小提琴图表示密度，黑色横线和粗竖线表示中位数与四分位范围，散点显示抽样窗口。四种方法的中位数都略低于 0：Mean 为 -0.120，Forward 为 -0.018，Linear 为 -0.019，Backward 为 -0.012，同时各方法都有明显长尾。

### 图 4.3-5 代表性预测窗口案例

文件：`paper_figures_4_3/forecast_cases/fig4_3_5_representative_forecast_case.{png,svg}`。

该图展示 Sundial 在 exchange_rate 数据集、30% 缺失率、medium term、window 13 上的代表性预测案例，预测区间为 2008-04-24 到 2009-02-17。上图比较 ground truth、clean input、Forward、Linear、Kalman-ARIMA 的预测轨迹；下图给出同一窗口的预测 sMAPE。Kalman-ARIMA 最低，为 0.019；clean input 为 0.037；Forward 为 0.041；Linear 为 0.060。

### 图 4.3-6 结构条件下的填补影响

文件：`paper_figures_4_3/structure_conditions/fig4_3_6_structure_condition_gain.{png,svg}`。

该图按完整结构差异评分的十分位分析填补影响。横轴是结构差异分位，纵轴是相对 clean 的 sMAPE 变化，线条对应 Mean、Forward、Linear、GP-RBF、SAITS，阴影表示四分位范围。高结构差异分位，尤其第 10 分位，Mean、Forward、GP-RBF 的误差升高明显，SAITS 相对更平缓。

### 图 4.3-7 结构通道解释力热力图

文件：`paper_figures_4_3/structure_conditions/fig4_3_7_structure_channel_heatmap.{png,svg}`。

该图比较不同结构指标组合对相对 sMAPE 变化的解释力。行表示结构指标组合或去除某类指标后的组合，列是 Median Spearman、Overall Spearman、Linear R2。full 组合的数值约为 0.238、0.217、0.154，整体解释力较强；drop seasonal 和 drop frequency 也较高，说明多类结构指标联合使用时解释效果更稳定。

## 原 figures 目录迁入图组

原 `figures/` 已迁入 `section_4_2_original/`。该图组是早期 4.2 汇总图，文件名仍保留原编号，目录隔离后不会与当前论文候选图编号冲突。

`section_4_2_original/section_4_2/fig4_2_1_model_imputer_heatmap.{png,svg}` 展示不同模型与填补方法组合下的 MSE 相对增幅中位数。它与当前图 4.2-3 左侧热力图表达接近，适合保留为早期中文版本或对照图。

`section_4_2_original/section_4_2/fig4_2_2_missing_ratio_distribution.{png,svg}` 展示缺失率从 10%、20% 到 30% 时预测误差变化的分布。箱线图和中位数连线显示 MSE 相对增幅中位数从 2.2% 升至 4.8% 和 8.8%，同时分布范围随缺失率升高而扩大。

`section_4_2_original/section_4_2/fig4_2_3_error_relation.{png,svg}` 并列展示点级重构误差与下游误差、结构扰动与下游误差的关系。图中给出分组 Spearman 中位数和正相关组占比，数值与当前图 4.2-6 一致，适合作为中文早期版本或验证材料。

## 批量窗口级诊断图

`outputs_by_model/<model>/<prediction_mode>_window_gap_scatter.{png,csv}` 是按模型与填补方法生成的窗口级散点图。一张图中包含该模型、该填补方法下的多个数据集，点颜色表示数据集。横轴是 history 与 prediction 的分布差异，纵轴是该填补方法相对 clean 的 sMAPE 差值。

`outputs_by_dataset/<model>/<dataset>_window_gap_scatter_by_imputation.{png,csv}` 是按模型与数据集生成的窗口级散点图。一张图中包含该模型、该数据集下的多种填补方法，点颜色表示填补方法。横轴和纵轴定义与 `outputs_by_model/` 相同。

这两类图更适合用于复查、附录或定位异常窗口。主文图建议优先使用 `paper_figures_4_2/` 与 `paper_figures_4_3/`。

## 数据来源与复现

4.2 候选图由 `paper_figures_4_2/generate_figures_4_2.py` 生成，主要读取 `results_analysis/*0509.csv`，并在 `paper_figures_4_2/data/` 保存扁平化记录、模型稳健性汇总、方法偏好矩阵、缺失率轨迹、数据集敏感性、结构解释和预测结构偏移等派生数据。

4.3 候选图由 `paper_figures_4_3/generate_figures_4_3.py` 生成，主要读取 `results_analysis/ablation/ablation_*.csv`、`outputs_by_dataset/*/*.csv`、`results_analysis/structure_metric_ablation/*.csv` 和 `tools/Sample/sample_forward_linear_kalman_forecast_*.csv`，并在 `paper_figures_4_3/data/` 保存窗口级点、误差分布、代表性案例和结构条件分析数据。

在仓库根目录运行以下命令可重新生成候选图：

```bash
python draw/paper_figures_4_2/generate_figures_4_2.py
python draw/paper_figures_4_3/generate_figures_4_3.py
```

早期批量诊断图脚本仍保留：

```bash
python draw/visualized_results_by_mode.py --model chronos2 --prediction_mode backward
python draw/visualized_results_by_dataset.py --model chronos2
```
