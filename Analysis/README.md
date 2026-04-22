# Analysis 模块说明

`analysis` 目录提供窗口级别的特征分析工具，能够针对以下三类数据批量计算 6 个基于 STL 的时间序列特征指标：

1. **预测窗口**：`datasets/Intermediate_Predictions` 中的各个预测结果片段。
2. **干净历史窗口**：原始 `datasets/ori` 数据集中与预测窗口相邻的历史部分。
3. **填补历史窗口**：`datasets/Imputed` 中不同缺失注入与填补策略生成的序列。

---

## 一、目录结构适配

### 预测窗口目录结构

模块支持两种预测窗口目录结构：

#### 1. 干净数据预测窗口
```
datasets/Intermediate_Predictions/
└── {dataset}_clean_{term}_prediction/
    ├── {dataset}_clean_{term}_prediction_0.csv
    ├── {dataset}_clean_{term}_prediction_1.csv
    └── ...
```

#### 2. 填补数据预测窗口（按填补方法分目录）
```
datasets/Intermediate_Predictions/
└── {dataset}_{method}_length{length}_{ratio}_{term}_prediction/
    ├── linear/                              # 填补方法子目录
    │   ├── {dataset}_{method}_..._prediction_0.csv
    │   └── ...
    ├── mean/
    │   └── ...
    └── forward/
        └── ...
```

---

## 二、使用方式

### 1. 单次分析脚本 `analysis/window_analysis.py`

该脚本通过 `mode` 子命令区分分析对象，可在任意目录使用 `python -m analysis.window_analysis` 调用。

#### (1) 预测窗口
- **命令示例**（必须单行）：
  - 干净数据：`python -m analysis.window_analysis prediction --prediction_dir datasets/Intermediate_Predictions/ETTh1_clean_long_prediction --output results/example_prediction.json`
  - 填补数据：`python -m analysis.window_analysis prediction --prediction_dir datasets/Intermediate_Predictions/ETTh1_BM_length50_010_long_prediction/linear --output results/example_prediction.json`
- **主要参数**
  - `--prediction_dir`：预测窗口所在目录。对于填补数据，路径应指向填补方法子目录。
  - `--output`（可选）：保存 JSON 结果的路径。

#### (2) 干净历史窗口
- **命令示例**：
  - `python -m analysis.window_analysis history --dataset ETTh1 --term long --output results/example_clean_history.json`
- **主要参数**
  - `--dataset`：数据集名称（需出现在 `datasets/dataset_properties.json`）。
  - `--term`：预测周期，支持 `short|medium|long`，默认 `short`。
  - `--output`：可选输出路径。

#### (3) 填补历史窗口
`window_analysis.py` 中提供 `analyze_imputed_history_windows` 函数，若需要独立脚本，可仿照 `history` 子命令调用；日常推荐使用下述批量脚本统一跑完所有填补方案。

### 2. 批量分析脚本 `analysis/batch_window_analysis.py`

`batch_window_analysis.py` 会自动遍历所有已生成的数据，按顺序完成预测窗口、干净历史、填补历史分析，并把结果写到 `results/window_analysis`。

- **基础命令**（单行）：
  - `python -m analysis.batch_window_analysis`

- **常用参数**
  | 参数 | 说明 |
  | --- | --- |
  | `--output_dir` | 结果保存根目录，默认 `results/window_analysis` |
  | `--datasets` | 逗号分隔的指定数据集（如 `ETTh1,ETTh2`） |
  | `--terms` | 逗号分隔的 term（如 `short,long`） |
  | `--impute_methods` | 逗号分隔的填补方法（如 `linear,mean`） |
  | `--skip_predictions` | 跳过预测窗口分析 |
  | `--skip_clean_history` | 跳过干净历史分析 |
  | `--skip_imputed_history` | 跳过填补历史分析 |

执行完成后会在 `results/window_analysis` 生成：
- `predictions/`：预测窗口分析结果，文件名格式：
  - 干净数据：`{dataset}_clean_{term}.json`
  - 填补数据：`{dataset}_{method}_{ratio}_{term}_{imputation_method}.json`
- `clean_history/`：干净历史窗口分析结果
- `imputed_history/`：填补历史窗口分析结果
- `summary.json`：统计整体运行概览

### 3. 输出格式

所有分析函数默认写出 **JSON**，其结构包含：
- 元信息：数据集、term、缺失/填补配置、周期等。
- `imputation_method`：填补方法名称（仅填补数据预测窗口有此字段）。
- `summary`：对 6 个指标的 `mean/std/min/max`。
- `window_results`：每个窗口的指标与索引信息。如果需要 CSV，可在读取 JSON 后视需要再转换为平面表格格式。

---

## 三、指标说明

以下指标均基于单变量序列的 STL 分解结果 `(trend, seasonal, residual)` 与原序列：

1. **趋势强度 (trend_strength)**  
   - 公式：`F_TS = max(0, 1 - Var(R) / Var(T + R))`  
   - 意义：衡量趋势在信号中占比，越接近 1 趋势越显著。

2. **趋势线性度 (trend_linearity)**  
   - 方法：将趋势分量拟合为正交二次回归 `T_t = β0 + β1 * P1(t) + β2 * P2(t)`，取 `β1` 为线性度指标。  
   - 意义：捕捉总体斜率方向，>0 表示上升趋势，<0 表示下降趋势。

3. **季节强度 (seasonal_strength)**  
   - 公式：`F_SS = max(0, 1 - Var(R) / Var(S + R))`  
   - 意义：季节性分量的相对强度，范围 [0,1]。

4. **季节相关性 (seasonal_correlation)**  
   - 计算：截取整数个周期，将所有季节段两两计算皮尔逊相关系数并取平均。  
   - 意义：衡量不同季节段的一致性，范围 [-1,1]。

5. **残差一阶自相关 (residual_autocorr_lag1)**  
   - 公式：`F_RA = E[(R_t - R̄)(R_{t-1} - R̄)] / Var(R)`  
   - 意义：残差的记忆程度，越接近 0 表示残差越独立。

6. **谱熵 (spectral_entropy)**  
   - 计算：对均值去除后的原序列做 FFT，取功率谱密度（PSD）并计算 `Σ log(PSD)`。  
   - 意义：衡量频谱分布复杂度，值越大说明频率成分越分散。

这些指标在 `analysis/metrics.py` 中实现，`window_analysis.py` 和 `batch_window_analysis.py` 会在每个窗口对所有数值列取平均后写入结果。

---

## 四、注意事项
- 运行前确保 `datasets/dataset_properties.json` 存在且包含所需数据集的 `period` 与 `terms` 信息。
- 预测/填补目录命名需符合现有正则格式（例如 `dataset_method_length50_ratio_term_prediction`）。
- STL 分解对时间序列长度有要求，窗口中有效样本需不少于 `2 * period`，否则该列会被跳过。
- 填补数据预测窗口需要指向具体的填补方法子目录，或让批量脚本自动遍历。

如需扩展更多指标或导出其它格式，可在现有脚本基础上添加自定义逻辑。
