# run_sundial.py 使用说明

## 快速开始

```bash
# 1. 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TSFIA
cd /home/decadent/TSF-Imputation-Analysis

# 2. 一键批量评估（推荐）
python Eval/run_sundial.py batch --dataset ETTh1 --method MCAR --device cpu

# 3. 单个评估
python Eval/run_sundial.py single --eval_data_path datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv --device cpu
```

## 常用命令

### 一键批量评估（新增）

只需提供数据集名和注空模式，自动批量评估所有缺失率和 term 组合：

```bash
# 评估 ETTh1 + MCAR（默认缺失率：5%,10%,15%,20%,25%,30%）
python Eval/run_sundial.py batch --dataset ETTh1 --method MCAR --device cpu

# 自定义缺失率
python Eval/run_sundial.py batch --dataset ETTh1 --method MCAR \
  --missing_ratios 0.05,0.10,0.15 --device cpu

# 使用 GPU
python Eval/run_sundial.py batch --dataset exchange_rate --method BM --device cuda:0
```

**自动识别 term 限制**：
- `national_illness` 等 short 数据集：只评估 short term
- `ETTh1`, `exchange_rate` 等 med_long 数据集：评估 short/medium/long 三种 term

### 单个评估

#### 自动模式（推荐）

自动从 `datasets/ori/` 查找干净数据集：

```bash
# 评估单个缺失数据集
python Eval/run_sundial.py single --eval_data_path datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv --device cpu

# 使用 GPU 加速
python Eval/run_sundial.py single --eval_data_path datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv --device cuda:0
```

#### 指定模式

手动指定干净数据集路径和 term（评估数据集文件名不要求格式）：

```bash
# 指定干净数据集和 term
python Eval/run_sundial.py single   --eval_data_path datasets/ori/ETTh1.csv --clean_data_path datasets/ori/ETTh1.csv  --term short --device cpu

# 任意文件名的评估数据集（不要求命名格式）
python Eval/run_sundial.py single \
  --eval_data_path datasets/my_custom_eval.csv \
  --clean_data_path datasets/MCAR/MCAR_050/ETTh1_MCAR_050_short.csv \
  --term short \
  --device cpu
```

**注意**：指定模式下必须提供 `--term` 参数，否则报错。

### 查看帮助

```bash
# 查看所有可用选项
python Eval/run_sundial.py --help
```

## 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--eval_data_path` | 评估数据集路径（必需） | `datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv` |
| `--clean_data_path` | 干净数据集路径（可选） | `datasets/MCAR/MCAR_050/ETTh1_MCAR_050_short.csv` |
| `--term` | 预测周期（指定模式下必需） | `short`, `medium`, `long` |
| `--device` | 运行设备 | `cpu`, `cuda:0`, `cuda:1` |
| `--batch_size` | 批次大小（默认：32） | `64`, `128` |
| `--debug` | 是否输出调试表格（默认：True） | `True`, `False` |
| `--output_dir` | 结果输出目录（默认：results/sundial） | `results/my_eval` |

## 数据集命名格式

评估数据集必须遵循以下命名格式：

```
[original_name]_method_ratio_term.csv
```

- `original_name`: 原始数据集名（如 `ETTh1`, `exchange_rate`）
- `method`: 缺失注入方法（`MCAR`, `BM`, `TM`, `TVMR`）
- `ratio`: 缺失比例（3 位数字，如 `005` 表示 5%）
- `term`: 预测周期（`short`, `medium`, `long`）

**示例**：
- `ETTh1_MCAR_005_long.csv`
- `exchange_rate_MCAR_005_medium.csv`
- `national_illness_BM_010_short.csv`

## 输出结果

**保存位置**: `results/sundial/`

**文件命名**: `{数据集名}_results.csv`

**示例**：
```bash
# 输入
datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv

# 输出
results/sundial/sundial_Missing/ETTh1_MCAR_005_short_results.csv
```

**输出格式** (CSV):
```csv
metric,value
eval_data,datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv
clean_data,datasets/ori/ETTh1.csv
freq,H
term,short
prediction_length,48
windows,10
MSE[mean],4.801281
MSE[0.5],4.860023
MAE[0.5],1.305689
MASE[0.5],0.888378
MAPE[0.5],0.231977
sMAPE[0.5],0.261466
MSIS,12.124096
RMSE[mean],2.191183
NRMSE[mean],0.363955
ND[0.5],0.216875
mean_weighted_sum_quantile_loss,0.182255
```

## 评估指标说明

| 指标 | 名称 | 说明 |
|------|------|------|
| MSE | 均方误差 | 预测值与真实值差的平方的平均 |
| MAE | 平均绝对误差 | 预测值与真实值差的绝对值的平均 |
| MASE | 平均绝对缩放误差 | 相对于基准方法的 MAE 比率 |
| MAPE | 平均绝对百分比误差 | 相对误差的百分比形式 |
| sMAPE | 对称 MAPE | 对称版本的 MAPE |
| MSIS | 修正的缩放区间得分 | 综合考虑准确性和区间宽度 |
| RMSE | 均方根误差 | MSE 的平方根 |
| NRMSE | 标准化 RMSE | 归一化的 RMSE |
| ND | 正态化偏差 | 归一化的偏差 |

## 运行示例

### 示例 1：评估 ETTh1 数据集（5% 缺失率）

```bash
python Eval/run_sundial.py \
  --eval_data_path datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv \
  --device cpu
```

### 示例 2：批量评估多个缺失率

```bash
python Eval/run_sundial.py \
  --eval_data_path \
    datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv \
    datasets/MCAR/MCAR_010/ETTh1_MCAR_010_short.csv \
    datasets/MCAR/MCAR_015/ETTh1_MCAR_015_short.csv \
  --device cpu
```

### 示例 3：使用 GPU 并关闭调试输出

```bash
python Eval/run_sundial.py \
  --eval_data_path datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv \
  --device cuda:0 \
  --debug False
```

## 注意事项

1. **数据集命名**：
   - 自动模式：评估数据集必须遵循 `[original_name]_method_ratio_term.csv` 格式
   - 指定模式：评估数据集文件名任意，不要求格式
2. **term 参数**：
   - 自动模式：从文件名解析，`--term` 参数被忽略（会显示警告）
   - 指定模式：必须提供 `--term` 参数（`short`, `medium`, `long` 三选一）
3. **干净数据集**：
   - 自动模式：从 `datasets/ori/{original_name}.csv` 自动查找
   - 指定模式：使用 `--clean_data_path` 手动指定（如 `datasets/MCAR/MCAR_050/xxx.csv`）
4. **频率**：自动从 `datasets/dataset_properties.json` 获取
5. **滑动窗口**：自动计算（最多 20 个窗口）
6. **GPU 内存**：如果显存不足，减小 `--batch_size`
