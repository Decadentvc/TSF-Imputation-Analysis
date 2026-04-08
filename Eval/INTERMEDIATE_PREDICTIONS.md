# 中间预测结果保存功能说明

## 概述

在评估过程中，系统会自动保存每个预测窗口的中间预测结果，方便后续分析和调试。

## 功能特性

1. **按窗口拆分保存**：每个预测窗口的结果保存为独立文件
2. **分离保存统计量**：mean 和 median 分别保存为两个独立文件
3. **包含时间序列**：每个文件不仅包含预测值，还包含对应的时间序列
4. **自动创建目录结构**：根据数据集名称自动创建子目录
5. **单次和批量评估均支持**：两种评估模式都会自动保存中间结果

## 保存路径

```
datasets/Intermediate_Predictions/
├── ETTh1/
│   ├── ETTh1_MCAR_005_short_prediction[mean]_0.csv
│   ├── ETTh1_MCAR_005_short_prediction[mean]_1.csv
│   ├── ETTh1_MCAR_005_short_prediction[0.5]_0.csv
│   ├── ETTh1_MCAR_005_short_prediction[0.5]_1.csv
│   └── ...
├── ETTh2/
│   └── ...
└── exchange_rate/
    └── ...
```

## 文件命名格式

### mean 预测结果

```
[eval_data_name]_prediction[mean]_[window_index].csv
```

示例：
- `ETTh1_MCAR_005_short_prediction[mean]_0.csv`
- `ETTh1_MCAR_005_short_prediction[mean]_1.csv`

### median 预测结果

```
[eval_data_name]_prediction[0.5]_[window_index].csv
```

示例：
- `ETTh1_MCAR_005_short_prediction[0.5]_0.csv`
- `ETTh1_MCAR_005_short_prediction[0.5]_1.csv`

## 文件内容格式

每个 CSV 文件包含两列：

| 列名 | 说明 |
|------|------|
| `date` | 时间序列（从预测窗口的起始时间开始） |
| `prediction` | 预测值（mean 或 median） |

示例：
```csv
date,prediction
2018-05-17 20:00:00,13.536441
2018-05-17 21:00:00,13.311673
2018-05-17 22:00:00,13.114555
...
```

## 使用方法

### 单次评估

```bash
python Eval/run_sundial.py single \
    --eval_data_path datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv \
    --imputation_method none \
    --device cpu
```

评估完成后，中间预测结果会自动保存到：
```
datasets/Intermediate_Predictions/ETTh1/ETTh1_MCAR_005_short_prediction_*.csv
```

### 批量评估

```bash
python Eval/run_sundial.py batch \
    --dataset ETTh1 \
    --method MCAR \
    --missing_ratios 0.05,0.10 \
    --imputation_methods none,zero,mean
```

批量评估会为每个评估数据集和每个填补方法都保存中间预测结果。

## 输出示例

```
================================================================================
Saving Intermediate Predictions
================================================================================
  Dataset: ETTh1
  Eval data: ETTh1_MCAR_005_short
  Prediction length: 48
  Number of windows: 20

  Sample output (window 0):
    Mean file: datasets/Intermediate_Predictions/ETTh1/ETTh1_MCAR_005_short_prediction[mean]_0.csv
    Median file: datasets/Intermediate_Predictions/ETTh1/ETTh1_MCAR_005_short_prediction[0.5]_0.csv
    Shape: (48, 2)
    Mean range: [10.5573, 13.7013]
    Median range: [10.9443, 13.6023]
    Date range: 2018-05-17 20:00:00 to 2018-05-19 19:00:00

  ✅ Saved 20 × 2 prediction files to: datasets/Intermediate_Predictions/ETTh1
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
- ETTh1 (频率: H, term: medium) → prediction_length = 96
- ETTh1 (频率: H, term: long) → prediction_length = 336

### 窗口数量

窗口数量由数据集长度和预测长度决定，计算公式：
```
windows = (dataset_length - prediction_length) // prediction_length
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
2. **文件覆盖**：相同评估数据集的中间结果会被覆盖
3. **性能影响**：保存中间结果对评估性能影响很小
4. **数据完整性**：每个窗口的预测长度固定，不会因为边界而变化

## 相关文件

- [run_sundial.py](file:///home/decadent/TSF-Imputation-Analysis/Eval/run_sundial.py) - 评估脚本（包含保存逻辑）
- [eval_sundial.py](file:///home/decadent/TSF-Imputation-Analysis/Eval/eval_sundial.py) - 核心评估模块（返回预测结果）
- [dataset_properties.json](file:///home/decadent/TSF-Imputation-Analysis/datasets/dataset_properties.json) - 数据集属性配置

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

### 代码位置

- 保存函数：`run_sundial.save_intermediate_predictions()`
- 调用位置：
  - 单次评估：`run_evaluation()` 函数末尾
  - 批量评估：`batch_evaluate()` 函数中的评估循环

---

**创建时间**：2026-04-08
**版本**：1.0
**作者**：AI Assistant
