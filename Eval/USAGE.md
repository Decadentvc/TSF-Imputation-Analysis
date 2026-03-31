# run_sundial.py 使用说明

## 1. 命令和示例

```bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TSFIA
cd /home/decadent/TSF-Imputation-Analysis
```

```bash
# 查看可用数据集
python Eval/run_sundial.py --list_datasets

# 跑单个数据集（自动选择 term）
python Eval/run_sundial.py --dataset ETTh1 --device cpu

# 跑所有数据集
python Eval/run_sundial.py --device cpu

# 指定 term
python Eval/run_sundial.py --dataset ETTh1 --term short --device cpu
python Eval/run_sundial.py --dataset ETTh1 --term medium --device cpu
python Eval/run_sundial.py --dataset ETTh1 --term long --device cpu

# 用 GPU
python Eval/run_sundial.py --dataset ETTh1 --device cuda:0

# 调整批次大小
python Eval/run_sundial.py --dataset ETTh1 --batch_size 64 --device cpu
```

## 2. 评估指标

输出包含以下指标：
- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **MASE**: 平均绝对缩放误差
- **MAPE**: 平均绝对百分比误差
- **sMAPE**: 对称平均绝对百分比误差
- **MSIS**: 修正的缩放区间得分
- **RMSE**: 均方根误差
- **NRMSE**: 标准化均方根误差
- **ND**: 正态化偏差
- **mean_weighted_sum_quantile_loss**: 分位数损失加权和

## 3. 输出结果

**保存位置**: `results/sundial/`

**文件命名**: `{数据集名}_{term}_results.csv`

**示例文件**: `ETTh1_short_results.csv`

**文件格式** (CSV):
```csv
metric,value
dataset,ETTh1
freq,H
term,short
prediction_length,48
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
