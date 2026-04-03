# 评估结果可视化使用说明

## 功能简介

`visualize_results.py` 脚本用于批量处理和可视化 Sundial 模型的评估结果。

主要功能：
1. 自动读取指定数据集和注空模式的所有结果文件（18 个文件）
2. 按 term（short/medium/long）分组整合数据
3. 生成汇总表格（CSV 格式）
4. 绘制指标随缺失率变化的折线图

## 快速开始

```bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TSFIA
cd /home/decadent/TSF-Imputation-Analysis

# 基本用法
python Eval/visualize_results.py --dataset ETTh1 --method MCAR
```

## 命令行参数

### 必需参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--dataset` | 数据集名称 | `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `exchange_rate`, `electricity`, `traffic`, `weather`, `national_illness` |
| `--method` | 注空模式 | `MCAR`, `BM`, `TM`, `TVMR` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--metrics` | `MSE[mean]`, `MAE[0.5]`, `MASE[0.5]` | 要绘制的指标列表 |
| `--results_dir` | `results/sundial/sundial_Missing` | 结果文件目录 |
| `--output_dir` | `results/sundial/visualization` | 输出目录 |
| `--plot_mode` | `both` | 绘图模式：`separate`（分 term 绘图）, `combined`（合并绘图）, `both`（两种都画） |

## 使用示例

### 1. 基本用法（默认指标）

```bash
python Eval/visualize_results.py --dataset ETTh1 --method MCAR
```

**输出**：
- `ETTh1_MCAR_summary.csv` - 汇总表格
- `ETTh1_MCAR_metrics.png` - 分 term 的三合一图表（short/medium/long 各一幅）
- `ETTh1_MCAR_MSE_mean__all_terms.png` - MSE[mean] 指标（三个 term 合并）
- `ETTh1_MCAR_MAE_0.5__all_terms.png` - MAE[0.5] 指标（三个 term 合并）
- `ETTh1_MCAR_MASE_0.5__all_terms.png` - MASE[0.5] 指标（三个 term 合并）

### 2. 自定义指标

```bash
# 指定要绘制的指标
python Eval/visualize_results.py --dataset ETTh1 --method MCAR \
  --metrics "MSE[mean]" "MAE[0.5]" "MAPE[0.5]" "sMAPE[0.5]"
```

**可用指标**：
- `MSE[mean]` - 均方误差（均值）
- `MSE[0.5]` - 均方误差（中位数）
- `MAE[0.5]` - 平均绝对误差
- `MASE[0.5]` - 平均绝对标度误差
- `MAPE[0.5]` - 平均绝对百分比误差
- `sMAPE[0.5]` - 对称平均绝对百分比误差
- `MSIS` - 平均缩放间隔得分
- `RMSE[mean]` - 均方根误差
- `NRMSE[mean]` - 归一化均方根误差

### 3. 只生成分 term 图表

```bash
python Eval/visualize_results.py --dataset ETTh1 --method MCAR \
  --plot_mode separate
```

**输出**：只生成 `ETTh1_MCAR_metrics.png`（3 个子图，每行一个 term）

### 4. 只生成合并图表

```bash
python Eval/visualize_results.py --dataset ETTh1 --method MCAR \
  --plot_mode combined
```

**输出**：为每个指标生成一张图（三个 term 在同一图中，不同颜色）

### 5. 批量处理多个数据集

```bash
# Bash 脚本批量处理
for dataset in ETTh1 ETTh2 ETTm1 ETTm2; do
    python Eval/visualize_results.py --dataset $dataset --method MCAR
done

# 处理所有注空模式
for method in MCAR BM TM TVMR; do
    python Eval/visualize_results.py --dataset ETTh1 --method $method
done
```

## 输出文件说明

### 1. 汇总表格（CSV）

**文件名**：`{dataset}_{method}_summary.csv`

**格式**：
```csv
Term,Missing Ratio,MSE[mean],MAE[0.5],MASE[0.5]
short,5%,6.311867,1.288026,0.828631
short,10%,6.401964,1.312030,0.843148
short,15%,6.428066,1.304454,0.841135
...
medium,5%,8.518450,1.749529,1.240194
...
long,5%,8.991443,1.824100,1.348003
...
```

**说明**：
- 每行对应一个 term 和一个缺失率的组合
- 包含所有指定指标的数值

### 2. 分 term 图表（PNG）

**文件名**：`{dataset}_{method}_metrics.png`

**布局**：
- 3 行 × N 列（N = 指标数量，默认 3）
- 每行对应一个 term（short/medium/long）
- 每列对应一个指标

**特点**：
- 横轴：缺失率（5%, 10%, 15%, 20%, 25%, 30%）
- 纵轴：指标值
- 每个子图显示该 term 下指标随缺失率的变化趋势
- 自动标注数据点数值

### 3. 合并图表（PNG）

**文件名**：`{dataset}_{method}_{metric}_all_terms.png`

**布局**：
- 单张图表
- 三条折线（红色=short, 绿色=medium, 蓝色=long）
- 便于对比不同 term 下的指标差异

## 结果文件命名规范

可视化脚本会自动识别以下命名格式的结果文件：

```
{dataset}_{method}_{ratio}_{term}_results.csv
```

**示例**：
- `ETTh1_MCAR_005_short_results.csv`
- `ETTh1_MCAR_010_medium_results.csv`
- `ETTh1_MCAR_015_long_results.csv`

其中：
- `dataset`: 原始数据集名称（如 ETTh1, exchange_rate）
- `method`: 注空模式（MCAR, BM, TM, TVMR）
- `ratio`: 缺失率编码（005=5%, 010=10%, ..., 030=30%）
- `term`: 预测期限（short, medium, long）

## 常见问题

### Q1: 找不到结果文件？

**A**: 确保结果文件在正确的目录下（默认 `results/sundial/sundial_Missing`），并且文件名符合规范。

```bash
# 检查结果文件
ls results/sundial/sundial_Missing/ETTh1_MCAR_*_results.csv
```

### Q2: 某些 term 没有数据？

**A**: 某些数据集（如 national_illness）只允许 short term，因此不会有 medium/long 的结果。脚本会自动跳过缺失的 term。

### Q3: 如何修改图表样式？

**A**: 可以编辑 `visualize_results.py` 中的绘图函数：
- 修改颜色：调整 `colors` 字典
- 修改标记：调整 `markers` 字典
- 修改尺寸：调整 `figsize` 参数

### Q4: 如何添加新指标？

**A**: 直接在 `--metrics` 参数中添加指标名称（必须与结果文件中的 metric 列完全匹配）：

```bash
python Eval/visualize_results.py --dataset ETTh1 --method MCAR \
  --metrics "MSE[mean]" "MAE[0.5]" "MASE[0.5]" "ND[0.5]"
```

## 完整工作流示例

```bash
# 1. 运行批量评估
python Eval/run_sundial.py batch --dataset ETTh1 --method MCAR --device cpu

# 2. 等待评估完成（生成 18 个结果文件）

# 3. 可视化结果
python Eval/visualize_results.py --dataset ETTh1 --method MCAR

# 4. 查看生成的文件
ls -lh results/sundial/visualization/ETTh1_MCAR_*

# 5. 打开图片查看（需要图形界面）
xdg-open results/sundial/visualization/ETTh1_MCAR_metrics.png
```

## 技术细节

### 数据解析

脚本使用正则表达式解析结果文件名：

```python
pattern = r'^([^.]+?)_(MCAR|BM|TM|TVMR)_(\d{3})_(short|medium|long)_results\.csv$'
```

### 缺失率转换

文件名中的缺失率编码（如 005）转换为实际百分比：

```python
ratio = int(ratio_code) / 1000.0  # 005 -> 0.005
ratio_pct = int(ratio * 1000)     # 0.005 -> 5%
```

### 图表生成

使用 matplotlib 生成高质量图表（300 DPI）：

```python
plt.savefig(output_file, dpi=300, bbox_inches='tight')
```

## 更新日志

- **2024-04-03**: 初始版本
  - 支持 4 种注空模式（MCAR, BM, TM, TVMR）
  - 支持 9 个数据集
  - 默认 3 个指标（MSE[mean], MAE[0.5], MASE[0.5]）
  - 分 term 和合并两种绘图模式
