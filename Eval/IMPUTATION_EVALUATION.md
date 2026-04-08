# 缺失值填补评估功能说明

## 功能概述

本模块支持在模型评估前对数据的缺失值进行填补，评估不同填补方法对预测结果的影响。

## 主要特性

1. **多种填补方法支持**：
   - `none`: 不填补（默认）
   - `zero`: 零值填补
   - `mean`: 均值填补
   - `forward`: 前向填充
   - `backward`: 后向填充
   - `linear`: 线性插值
   - `nearest`: 最近邻插值
   - `spline`: 样条插值
   - `seasonal`: 季节性分解填补

2. **灵活的评估模式**：
   - 单次评估：对单个数据集使用指定填补方法
   - 批量评估：对多个数据集依次使用多种填补方法

3. **结果保存**：
   - 填补方法为"none"的结果保存到 `results/sundial/sundial_Ori` 目录
   - 其他填补方法的结果保存到 `results/sundial/sundial_Impute` 目录
   - 文件名格式：`[impute_method]_[ori_dataset_name]_[method]_[ratio]_[term]_result.csv`

## 使用方法

### 单次评估模式

```bash
# 使用零值填补进行评估
python Eval/run_sundial.py single --eval_data_path datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv --imputation_method zero

# 使用均值填补进行评估
python Eval/run_sundial.py single --eval_data_path datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv --imputation_method mean

# 不填补直接评估（默认）
python Eval/run_sundial.py single --eval_data_path datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv --imputation_method none
```

### 批量评估模式

```bash
# 使用所有填补方法批量评估（默认）
python Eval/run_sundial.py batch --dataset ETTh1 --method MCAR

# 指定部分填补方法进行评估
python Eval/run_sundial.py batch --dataset ETTh1 --method MCAR --imputation_methods none,zero,mean

# 指定缺失比例和填补方法
python Eval/run_sundial.py batch --dataset ETTh1 --method MCAR --missing_ratios 0.05,0.10,0.15 --imputation_methods zero,mean,linear
```

## 评估逻辑

1. **单次评估**：
   - 读取评估数据集（含缺失值）
   - 应用指定的填补方法
   - 使用填补后的数据进行预测
   - 与干净数据集对比计算指标
   - 保存结果

2. **批量评估**：
   - 对每个数据集文件：
     - 依次使用填补方法集合中的每种方法
     - 每次重新读取原始数据（不混用）
     - 应用当前填补方法
     - 进行评估
     - 保存结果到对应目录
   - 中间填补数据不保存，节省存储空间

## 输出文件命名示例

假设原始数据集为 `ETTh1_MCAR_005_short.csv`：

- 不填补：`results/sundial/sundial_Ori/ETTh1_MCAR_005_short_results.csv`
- 零值填补：`results/sundial/sundial_Impute/zero_ETTh1_MCAR_005_short_results.csv`
- 均值填补：`results/sundial/sundial_Impute/mean_ETTh1_MCAR_005_short_results.csv`
- 线性插值：`results/sundial/sundial_Impute/linear_ETTh1_MCAR_005_short_results.csv`

## 注意事项

1. 填补操作在内存中进行，不会保存中间填补后的数据集
2. 每次评估都会重新读取原始数据，确保不同填补方法之间不互相影响
3. 批量评估默认使用所有填补方法，可通过 `--imputation_methods` 参数自定义
4. 结果文件中的 `imputation_method` 字段记录了使用的填补方法
