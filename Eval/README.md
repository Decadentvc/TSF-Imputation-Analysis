# 中间预测结果保存功能说明

## 概述

在评估过程中，系统会自动保存每个预测窗口的中间预测结果，方便后续分析和调试。

## 功能特性

1. **按窗口拆分保存**：每个预测窗口的结果保存为独立文件
2. **只保存关键信息**：每个文件包含时序列和 mean 预测值
3. **按评估任务组织**：根据评估任务名称创建子目录，同一任务的所有窗口预测结果保存在同一目录
4. **按填补方法分类**：不同填补方法的预测结果保存到各自的子目录，避免互相覆盖
5. **自动创建目录结构**：根据评估数据集名称和填补方法自动创建子目录
6. **单次和批量评估均支持**：两种评估模式都会自动保存中间结果

## 保存路径

```
datasets/Intermediate_Predictions/
├── ETTh1_MCAR_005_short_prediction/
│   ├── linear/                              # 填补方法子目录
│   │   ├── ETTh1_MCAR_005_short_prediction_0.csv
│   │   ├── ETTh1_MCAR_005_short_prediction_1.csv
│   │   └── ...
│   ├── mean/
│   │   ├── ETTh1_MCAR_005_short_prediction_0.csv
│   │   └── ...
│   └── forward/
│       └── ...
├── ETTh1_BM_010_long_prediction/
│   ├── linear/
│   │   └── ...
│   └── mean/
│       └── ...
├── ETTh1_clean_short_prediction/            # 干净数据（无填补方法子目录）
│   ├── ETTh1_clean_short_prediction_0.csv
│   ├── ETTh1_clean_short_prediction_1.csv
│   └── ...
└── exchange_rate_MCAR_005_medium_prediction/
    └── ...
```

## 目录结构

### 填补数据预测结果

每个评估任务创建一个子目录，并按填补方法再分目录：

```
[eval_data_name]_prediction/[imputation_method]/
```

示例：
- `ETTh1_MCAR_005_short_prediction/linear/` - ETTh1 数据集，MCAR 模式，5% 缺失率，short term，linear 填补的预测结果
- `ETTh1_BM_010_long_prediction/mean/` - ETTh1 数据集，BM 模式，10% 缺失率，long term，mean 填补的预测结果

### 干净数据预测结果

干净数据评估不包含填补方法子目录：

```
[dataset_name]_clean_[term]_prediction/
```

示例：
- `ETTh1_clean_short_prediction/` - ETTh1 干净数据，short term 的预测结果

## 文件命名格式

```
[eval_data_name]_prediction_[window_index].csv
```

示例：
- `ETTh1_MCAR_005_short_prediction_0.csv` - 第 0 个预测窗口
- `ETTh1_MCAR_005_short_prediction_1.csv` - 第 1 个预测窗口
- `ETTh1_MCAR_005_short_prediction_2.csv` - 第 2 个预测窗口

## 文件内容格式

每个 CSV 文件包含两列：

| 列名 | 说明 |
|------|------|
| `date` | 时间序列（从预测窗口的起始时间开始） |
| `prediction` | 预测值（mean） |

示例：
```csv
date,prediction
2018-05-17 20:00:00,13.536441
2018-05-17 21:00:00,13.311673
2018-05-17 22:00:00,13.114555
...
```

## 使用方法

### 批量评估（推荐）

使用 `eval_clean_vs_imputed.py` 进行批量评估：

```bash
# 评估所有数据集，使用所有填补方法
python eval/eval_clean_vs_imputed.py --all_datasets --method BM --imputation_methods all --device cuda

# 评估所有数据集，指定填补方法
python eval/eval_clean_vs_imputed.py --all_datasets --method BM --imputation_methods linear,mean,forward,backward --device cuda

# 评估单个数据集
python eval/eval_clean_vs_imputed.py --dataset ETTh1 --method BM --imputation_methods linear,mean --device cuda

# 指定缺失率
python eval/eval_clean_vs_imputed.py --all_datasets --method BM --missing_ratios 0.10,0.20,0.30 --imputation_methods linear --device cuda
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--all_datasets` | 遍历 dataset_properties.json 中的所有数据集 |
| `--dataset` | 指定单个数据集名称 |
| `--method` | 缺失模式（MCAR/BM/TM/TVMR） |
| `--missing_ratios` | 缺失率列表，逗号分隔 |
| `--imputation_methods` | 填补方法列表，逗号分隔；使用 `all` 表示全部 |
| `--device` | 推理设备（cuda/cpu） |
| `--force` | 强制重新计算，覆盖已有结果 |

### 可用填补方法

| 方法 | 说明 |
|------|------|
| `none` | 不填补（保留缺失值） |
| `zero` | 零值填补 |
| `mean` | 均值填补 |
| `forward` | 前向填充 |
| `backward` | 后向填充 |
| `linear` | 线性插值 |
| `nearest` | 最近邻插值 |
| `spline` | 样条插值 |
| `seasonal` | 季节性分解填补 |

## 输出示例

```
================================================================================
Saving Intermediate Predictions
================================================================================
  Dataset: ETTh1
  Eval data: ETTh1_MCAR_005_short
  Imputation method: linear
  Prediction length: 48
  Frequency: H
  Number of windows: 20

  Sample output (window 0):
    File: datasets/Intermediate_Predictions/ETTh1_MCAR_005_short_prediction/linear/ETTh1_MCAR_005_short_prediction_0.csv
    Shape: (48, 2)
    Mean range: [10.5573, 13.7013]
    Date range: 2018-05-17 20:00:00 to 2018-05-19 19:00:00

  ✅ Saved 20 prediction windows to: datasets/Intermediate_Predictions/ETTh1_MCAR_005_short_prediction/linear
================================================================================
```

## 参数说明

### 预测长度

每个文件的行数等于预测长度（prediction_length），由以下因素决定：
- **数据频率**：从 `dataset_properties.json` 读取
- **term 类型**：short/medium/long
- **计算公式**：`prediction_length = base_length × term_multiplier`

示例：
- ETTh1 (频率: H, term: short) → prediction_length = 48
- ETTh1 (频率: H, term: medium) → prediction_length = 480
- ETTh1 (频率: H, term: long) → prediction_length = 720

### 窗口数量

窗口数量由数据集长度和预测长度决定，计算公式：
```
windows = min(max(1, ceil(0.6 * dataset_length / prediction_length)), 20)
```

示例：
- ETTh1 short: 20 个窗口
- ETTh1 medium: 约 180 个窗口
- ETTh1 long: 约 49 个窗口

## 应用场景

1. **预测结果分析**：查看每个窗口的预测值分布
2. **异常检测**：识别预测异常的窗口
3. **方法对比**：比较不同填补方法对预测结果的影响
4. **调试辅助**：定位评估过程中的问题

## 注意事项

1. **存储空间**：批量评估会产生大量文件，请确保有足够的存储空间
2. **目录隔离**：不同填补方法的预测结果保存在不同子目录，不会互相覆盖
3. **性能影响**：保存中间结果对评估性能影响很小
4. **数据完整性**：每个窗口的预测长度固定，不会因为边界而变化

## 相关文件

- [run_sundial.py](file:///d:/Projects/GitHub/TSF-Imputation-Analysis/eval/run_sundial.py) - 评估脚本（包含保存逻辑）
- [eval_sundial.py](file:///d:/Projects/GitHub/TSF-Imputation-Analysis/eval/eval_sundial.py) - 核心评估模块（返回预测结果）
- [eval_clean_vs_imputed.py](file:///d:/Projects/GitHub/TSF-Imputation-Analysis/eval/eval_clean_vs_imputed.py) - 统一评估脚本
- [dataset_properties.json](file:///d:/Projects/GitHub/TSF-Imputation-Analysis/datasets/dataset_properties.json) - 数据集属性配置

## 技术细节

### 实现原理

1. **预测结果获取**：
   - `evaluate_sundial()` 返回包含 `forecasts` 的结果字典
   - 每个 `forecast` 是 `SampleForecast` 对象，包含多个采样结果

2. **统计量计算**：
   - `forecast.mean`：计算所有采样的均值
   - `forecast.quantile(0.5)`：计算 0.5 分位数（中位数）

3. **文件保存**：
   - 使用 pandas DataFrame 保存为 CSV
   - 每个窗口一个文件，便于单独分析
   - 按填补方法创建子目录，避免覆盖

### 代码位置

- 保存函数：`run_sundial.save_intermediate_predictions()`
- 调用位置：
  - `eval_clean_vs_imputed.py` 中的 `evaluate_imputed_datasets()` 函数
  - `run_sundial.py` 中的 `run_evaluation()` 函数
  - `run_sundial.py` 中的 `batch_evaluate()` 函数
  - `run_sundial.py` 中的 `evaluate_clean()` 函数

---

**更新时间**：2026-04-11
**版本**：2.0
