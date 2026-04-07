# 缺失值注入工具使用说明

## 目录结构

```
Missing_Value_Injection/for_sundial/
├── inject_range_utils.py    # 注错区间确定工具
├── MCAR.py                   # MCAR 缺失注入函数库
├── MI_all.py                 # 统一入口脚本（命令行工具）
└── README.md                 # 使用说明
```

## 快速使用

### 命令行工具

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TSFIA
cd /home/decadent/TSF-Imputation-Analysis

# 查看帮助
python Missing_Value_Injection/for_sundial/MI_all.py --help
```

## 使用方法

### 基本命令格式

```bash
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset <数据集名> \
  --missing_ratio <缺失比例> \
  [--term short|medium|long] \
  [--data_path datasets] \
  [--seed 42] \
  [--no_auto_term] \
  [--output_dir datasets/MI]
```

### 1. 批量缺失率注入

**支持单个缺失率**：
```bash
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio 0.05
```

**支持批量缺失率**（逗号分隔）：
```bash
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio "0.05,0.1,0.15"
```

**支持方括号格式**：
```bash
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio "[0.05,0.1,0.15]"
```

### 2. 自动 Term 检测

**自动根据数据集配置选择 term**（默认行为）：

```bash
# national_illness (term="short") -> 只注入 short
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset national_illness \
  --missing_ratio 0.05

# ETTh1 (term="med_long") -> 自动注入 short, medium, long 三种 term
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio 0.05
```

**手动指定 term**（禁用自动检测）：
```bash
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio 0.05 \
  --term short \
  --no_auto_term
```

## 完整命令示例

### 示例 1：单个缺失率，自动 term

```bash
# national_illness 只跑 short
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset national_illness \
  --missing_ratio 0.05

# ETTh1 自动跑 short, medium, long
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio 0.05
```

### 示例 2：批量缺失率，自动 term

```bash
# national_illness 只跑 short，注入 3 个缺失率
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset national_illness \
  --missing_ratio "0.05,0.1,0.15"

# ETTh1 自动跑 short, medium, long，每个 term 注入 3 个缺失率（共 9 个文件）
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio "0.05,0.1,0.15"
```

### 示例 3：手动指定 term

```bash
# 强制只跑 short term
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio 0.05 \
  --term short \
  --no_auto_term

# 强制只跑 medium term
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio 0.05 \
  --term medium \
  --no_auto_term
```

### 示例 4：自定义随机种子

```bash
python Missing_Value_Injection/for_sundial/MI_all.py \
  --missing_pattern MCAR \
  --dataset ETTh1 \
  --missing_ratio 0.05 \
  --seed 123
```

## 文件保存规则

### 命名格式

- **目录**：`datasets/MI/<pattern>_<ratio>/`
- **文件**：`<dataset>_<pattern>_<ratio>_<term>.csv`

### 单个 term 的数据集（如 national_illness）

```
datasets/MI/MCAR_005/national_illness_MCAR_005_short.csv
```

### 多个 term 的数据集（如 ETTh1）

```
datasets/MI/MCAR_005/ETTh1_MCAR_005_short.csv
datasets/MI/MCAR_005/ETTh1_MCAR_005_medium.csv
datasets/MI/MCAR_005/ETTh1_MCAR_005_long.csv
```

### 缺失率命名

| 缺失比例 | 命令行参数 | 目录名 | 文件名示例 |
|---------|-----------|--------|-----------|
| 5% | `--missing_ratio 0.05` | `MCAR_005` | `ETTh1_MCAR_005_short.csv` |
| 10% | `--missing_ratio 0.1` | `MCAR_010` | `ETTh1_MCAR_010_short.csv` |
| 15% | `--missing_ratio 0.15` | `MCAR_015` | `ETTh1_MCAR_015_short.csv` |

**注意**：文件名始终包含 term 信息，无论数据集有几个 term。

## 输出说明

### 示例输出（ETTh1, 5% MCAR, 自动 term）

```
缺失比例：5.00%
自动检测到数据集 'ETTh1' 的 term 配置：['short', 'medium', 'long']

获取数据集 'ETTh1' (short) 的注错区间...
  注错区间：[13580, 17372)
  注错区间长度：3792

================================================================================
注入：pattern=MCAR, term=short, missing_ratio=5.00%
================================================================================

注入结果:
  总单元格数：26544
  注入缺失值数：1327
  实际缺失比例：5.00%

文件保存:
  datasets/MI/MCAR_005/ETTh1_MCAR_005_short.csv
================================================================================

...

================================================================================
批量注入完成！共生成 3 个文件:
================================================================================
1. datasets/MI/MCAR_005/ETTh1_MCAR_005_short.csv
2. datasets/MI/MCAR_005/ETTh1_MCAR_005_medium.csv
3. datasets/MI/MCAR_005/ETTh1_MCAR_005_long.csv
```

## Python 代码调用

### 调用 MI_all.py（推荐）

```python
import sys
sys.path.insert(0, "Missing_Value_Injection/for_sundial")

from MI_all import run_mcar_injection

# 运行 MCAR 注入
results = run_mcar_injection(
    dataset_name="ETTh1",
    missing_ratios=[0.05, 0.1],
    terms=["short", "medium", "long"],
    data_path="datasets",
    output_base_dir="datasets/MI",
    seed=42
)

# 访问结果
for result in results:
    print(f"文件：{result['output_path']}")
    print(f"缺失值数：{result['injected_missing']}")
```

### 直接调用 MCAR.py（底层 API）

```python
from inject_range_utils import get_injection_range
from MCAR import inject_mcar

# 1. 获取注错区间
injection_range = get_injection_range("ETTh1", "short", "datasets")

# 2. 注入缺失值
df_injected, info = inject_mcar(
    dataset_name="ETTh1",
    injection_range=injection_range,
    missing_ratio=0.05,
    term="short",
    seed=42
)

# 3. 保存结果
df_injected.to_csv("output.csv", index=False)

# 4. 访问注入信息
print(f"注入缺失值数：{info['injected_missing']}")
print(f"实际缺失比例：{info['actual_missing_ratio']:.2%}")
```

## 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--missing_pattern` | 是 | - | 缺失模式（MCAR, MAR, MNAR） |
| `--dataset` | 是 | - | 数据集名称 |
| `--missing_ratio` | 是 | - | 缺失比例，支持单个值或逗号分隔列表 |
| `--term` | 否 | None | short/medium/long（默认自动检测） |
| `--data_path` | 否 | datasets | 数据集目录 |
| `--seed` | 否 | 42 | 随机种子 |
| `--no_auto_term` | 否 | False | 禁用自动 term 检测 |
| `--output_dir` | 否 | datasets/MI | 输出目录基础路径 |

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
  python Missing_Value_Injection/for_sundial/MI_all.py \
    --missing_pattern MCAR \
    --dataset $dataset \
    --missing_ratio 0.05
done
```

### 为单个数据集注入多个缺失率

```bash
python Missing_Value_Injection/for_sundial/MI_all.py --missing_pattern MCAR --dataset ETTh1 --missing_ratio "0.05,0.1,0.15,0.2,0.25,0.3"
```

## 注意事项

1. **自动 term 检测**：默认启用，根据 `dataset_properties.json` 配置自动选择
2. **批量注入**：支持多个缺失率，自动生成多个文件
3. **文件命名规范**：文件名始终包含 pattern、ratio 和 term 信息
4. **随机种子**：每个 term 和缺失率组合使用不同的种子，保证可复现性
5. **原始数据保护**：原始数据不会被修改
6. **格式灵活**：缺失率支持多种输入格式（`0.05`, `0.05,0.1`, `[0.05,0.1]`）
7. **精确缺失比例**：使用四舍五入 + 补偿机制，确保实际缺失比例与预期一致（误差 < 0.01%）
8. **架构说明**：
   - `MI_all.py`：统一入口，支持多种缺失模式（MCAR, MAR, MNAR）
   - `MCAR.py`：纯函数库，负责具体的 MCAR 缺失值注入逻辑
   - `inject_range_utils.py`：工具库，负责计算注错区间
9. **扩展性**：可轻松添加新的缺失模式（MAR, MNAR 等），只需创建对应的函数库并在 MI_all.py 中添加分支
