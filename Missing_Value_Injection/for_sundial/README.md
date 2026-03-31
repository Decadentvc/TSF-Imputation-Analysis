# MCAR 缺失值注入工具使用说明（智能版）

## 目录结构

```
Missing_Value_Injection/for_sundial/
├── inject_range_utils.py    # 注错区间确定工具（被各种注错模式调用）
└── MCAR.py                   # MCAR 完全随机缺失注入模式（智能版）
```

## 快速使用

### 命令行工具

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TSFIA
cd /home/decadent/TSF-Imputation-Analysis

# 查看帮助
python Missing_Value_Injection/for_sundial/MCAR.py --help
```

## 智能特性

### 1. 批量缺失率注入

**支持单个缺失率**：
```bash
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio 0.05
```

**支持批量缺失率**（逗号分隔）：
```bash
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio "0.05,0.1,0.15"
```

**支持方括号格式**：
```bash
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio "[0.05,0.1,0.15]"
```

### 2. 自动 Term 检测

**自动根据数据集配置选择 term**（默认行为）：

```bash
# national_illness (term="short") -> 只注入 short
python Missing_Value_Injection/for_sundial/MCAR.py --dataset national_illness --missing_ratio 0.05

# ETTh1 (term="med_long") -> 自动注入 short, medium, long 三种 term
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio 0.05
```

**手动指定 term**（禁用自动检测）：
```bash
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio 0.05 --term short --no_auto_term
```

## 完整命令示例

### 示例 1：单个缺失率，自动 term

```bash
# national_illness 只跑 short
python Missing_Value_Injection/for_sundial/MCAR.py --dataset national_illness --missing_ratio 0.05

# ETTh1 自动跑 short, medium, long
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio 0.05
```

### 示例 2：批量缺失率，自动 term

```bash
# national_illness 只跑 short，注入 3 个缺失率
python Missing_Value_Injection/for_sundial/MCAR.py --dataset national_illness --missing_ratio "0.05,0.1,0.15"

# ETTh1 自动跑 short, medium, long，每个 term 注入 3 个缺失率（共 9 个文件）
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio "0.05,0.1,0.15"
```

### 示例 3：手动指定 term

```bash
# 强制只跑 short term
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio 0.05 --term short --no_auto_term

# 强制只跑 medium term
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio 0.05 --term medium --no_auto_term
```

### 示例 4：自定义随机种子

```bash
python Missing_Value_Injection/for_sundial/MCAR.py --dataset ETTh1 --missing_ratio 0.05 --seed 123
```

## 文件保存规则

### 命名格式

- **目录**：`datasets/MCAR/MCAR_[ratio]/`
- **文件**：根据 term 数量自动调整

### 单个 term 的数据集（如 national_illness）

```
datasets/MCAR/MCAR_005/national_illness_MCAR_005.csv
```

### 多个 term 的数据集（如 ETTh1）

```
datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv
datasets/MCAR/MCAR_005/ETTh1_MCAR_005_medium.csv
datasets/MCAR/MCAR_005/ETTh1_MCAR_005_long.csv
```

### 缺失率命名

| 缺失比例 | 命令行参数 | 目录名 | 文件名示例 |
|---------|-----------|--------|-----------|
| 5% | `--missing_ratio 0.05` | `MCAR_005` | `ETTh1_MCAR_005_short.csv` |
| 10% | `--missing_ratio 0.1` | `MCAR_010` | `ETTh1_MCAR_010_short.csv` |
| 15% | `--missing_ratio 0.15` | `MCAR_015` | `ETTh1_MCAR_015_short.csv` |

## 输出说明

### 示例输出（ETTh1, 5% MCAR, 自动 term）

```
缺失比例：5.00%
自动检测到数据集 'ETTh1' 的 term 配置：['short', 'medium', 'long']

================================================================================
开始注入：term=short, missing_ratio=5.00%
================================================================================
================================================================================
MCAR 缺失值注入 - ETTh1 (short)
================================================================================
注入参数:
  缺失比例：5.00%
  注错区间：[13580, 17372)
  注错区间长度：3792

注入结果:
  总单元格数：26544
  注入缺失值数：1294
  实际缺失比例：4.87%

文件保存:
  原始文件：datasets/ori/ETTh1.csv
  注入后文件：datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv
================================================================================

================================================================================
开始注入：term=medium, missing_ratio=5.00%
================================================================================
...

================================================================================
批量注入完成！共生成 3 个文件:
================================================================================
1. datasets/MCAR/MCAR_005/ETTh1_MCAR_005_short.csv
2. datasets/MCAR/MCAR_005/ETTh1_MCAR_005_medium.csv
3. datasets/MCAR/MCAR_005/ETTh1_MCAR_005_long.csv
```

## Python 代码调用

```python
from Missing_Value_Injection.for_sundial.MCAR import inject_mcar

# 方式 1：单个缺失率，自动 term
results = inject_mcar(
    dataset_name="ETTh1",
    missing_ratio=0.05,
    auto_term=True
)

# 方式 2：批量缺失率，自动 term
results = inject_mcar(
    dataset_name="ETTh1",
    missing_ratio=[0.05, 0.1, 0.15],
    auto_term=True
)

# 方式 3：手动指定 term
results = inject_mcar(
    dataset_name="ETTh1",
    missing_ratio=0.05,
    term="short",
    auto_term=False
)

# 访问结果
for result in results:
    print(f"文件：{result['output_path']}")
    print(f"缺失值数：{result['injected_missing']}")
```

## 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | 是 | - | 数据集名称 |
| `--missing_ratio` | 是 | - | 缺失比例，支持单个值或逗号分隔列表 |
| `--term` | 否 | None | short/medium/long（默认自动检测） |
| `--data_path` | 否 | datasets | 数据集目录 |
| `--seed` | 否 | 42 | 随机种子 |
| `--no_auto_term` | 否 | False | 禁用自动 term 检测 |

## 数据集 term 配置

数据集的 term 配置在 `datasets/dataset_properties.json` 中定义：

- **term="short"**: 只注入 short term
  - national_illness
  
- **term="med_long"**: 自动注入 short, medium, long 三种 term
  - ETTh1, ETTh2, ETTm1, ETTm2
  - electricity, exchange_rate, traffic, weather

## 批量注入示例

### 为所有数据集注入 5% 缺失值

```bash
for dataset in ETTh1 ETTh2 ETTm1 ETTm2; do
  python Missing_Value_Injection/for_sundial/MCAR.py \
    --dataset $dataset --missing_ratio 0.05
done
```

### 为单个数据集注入多个缺失率

```bash
python Missing_Value_Injection/for_sundial/MCAR.py \
  --dataset ETTh1 --missing_ratio "0.05,0.1,0.15,0.2"
```

## 注意事项

1. **自动 term 检测**：默认启用，根据 `dataset_properties.json` 配置自动选择
2. **批量注入**：支持多个缺失率，自动生成多个文件
3. **文件命名**：多个 term 时文件名包含 term 后缀
4. **随机种子**：每个 term 和缺失率组合使用不同的种子，保证可复现性
5. **原始数据保护**：原始数据不会被修改
6. **格式灵活**：缺失率支持多种输入格式（`0.05`, `0.05,0.1`, `[0.05,0.1]`）
7. **精确缺失比例**：使用四舍五入 + 补偿机制，确保实际缺失比例与预期一致（误差 < 0.01%）
