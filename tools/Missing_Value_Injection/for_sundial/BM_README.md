# BM（块缺失）缺失值注入模块使用说明

## 概述

BM（Block Missing）模块实现了块缺失模式的缺失值注入功能。与 MCAR（完全随机缺失）不同，BM 模式下的缺失值以连续块的形式出现，更贴近实际应用场景中的设备故障、传感器失效等情况。

## 核心特性

1. **块状缺失**：缺失值以连续块的形式出现，而非随机分散
2. **固定块长度**：每个缺失块的长度固定（默认 50）
3. **不重叠不相邻**：块与块之间保证不重叠、不相邻
4. **只对非时序列注入**：时间列（date/time/timestamp）不会被注入缺失值

## 使用方法

### 1. 命令行使用

```bash
# 基本用法
python Missing_Value_Injection/for_sundial/MI_all.py \
    --missing_pattern BM \
    --dataset ETTh1 \
    --missing_ratio 0.05

# 指定块长度
python Missing_Value_Injection/for_sundial/MI_all.py \
    --missing_pattern BM \
    --dataset ETTh1 \
    --missing_ratio 0.10 \
    --block_length 100

# 批量生成多个缺失率
python Missing_Value_Injection/for_sundial/MI_all.py \
    --missing_pattern BM \
    --dataset ETTh1 \
    --missing_ratio 0.05,0.10,0.15,0.20 \
    --block_length 50
```

### 2. Python 代码调用

```python
from Missing_Value_Injection.for_sundial.BM import inject_bm
from Missing_Value_Injection.for_sundial.inject_range_utils import get_injection_range

# 获取注错区间
injection_range = get_injection_range(
    dataset_name="ETTh1",
    term="short",
    data_path="datasets"
)
injection_range["data_path"] = "datasets"

# 执行 BM 注入
df, info = inject_bm(
    dataset_name="ETTh1",
    injection_range=injection_range,
    missing_ratio=0.1,
    term="short",
    block_length=50,
    seed=42
)

# 查看注入信息
print(f"注入缺失值数: {info['injected_missing']}")
print(f"实际缺失比例: {info['actual_missing_ratio']:.2%}")
print(f"块数量: {info['n_blocks']}")
```

## 参数说明

### 命令行参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--missing_pattern` | str | 是 | - | 缺失模式，选择 "BM" |
| `--dataset` | str | 是 | - | 数据集名称（如 ETTh1, exchange_rate） |
| `--missing_ratio` | str | 是 | - | 缺失比例，支持逗号分隔（如 0.05,0.10） |
| `--term` | str | 否 | 自动 | 预测周期（short/medium/long） |
| `--block_length` | int | 否 | 50 | 块长度 |
| `--data_path` | str | 否 | datasets | 数据集目录 |
| `--output_dir` | str | 否 | datasets | 输出目录 |
| `--seed` | int | 否 | 42 | 随机种子 |
| `--no_auto_term` | flag | 否 | - | 禁用自动 term 检测 |

### inject_bm 函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_name` | str | - | 数据集名称 |
| `injection_range` | dict | - | 注错区间信息 |
| `missing_ratio` | float | - | 缺失比例（0.0-1.0） |
| `term` | str | - | term 类型 |
| `block_length` | int | 50 | 块长度 |
| `seed` | int | 42 | 随机种子 |

## 输出文件

### 文件命名格式

```
{output_dir}/BM/BM_{ratio}/{dataset}_BM_{ratio}_{term}.csv
```

示例：
```
datasets/BM/BM_005/ETTh1_BM_005_short.csv
datasets/BM/BM_010/ETTh1_BM_010_medium.csv
```

### 返回信息

`inject_bm` 函数返回一个元组 `(df, info)`：

- `df`: 注入缺失值后的 DataFrame
- `info`: 包含以下字段的字典：
  - `dataset_name`: 数据集名称
  - `term`: term 类型
  - `missing_ratio`: 目标缺失比例
  - `block_length`: 块长度
  - `n_blocks`: 块数量
  - `total_cells`: 注错区间总单元格数
  - `injected_missing`: 实际注入的缺失值数量
  - `actual_missing_ratio`: 实际缺失比例
  - `injection_range`: 注错区间信息
  - `block_positions`: 各块的位置信息
  - `data_columns`: 数据列列表

## 示例输出

```
================================================================================
注入：pattern=BM, term=short, missing_ratio=5.00%
  块长度：50
================================================================================

注入结果:
  总单元格数：3792
  块长度：50
  块数量：3
  注入缺失值数：150
  实际缺失比例：3.96%

文件保存:
  datasets/BM/BM_005/ETTh1_BM_005_short.csv
================================================================================
```

## 与 MCAR 的对比

| 特性 | MCAR | BM |
|------|------|-----|
| 缺失分布 | 随机分散 | 连续块状 |
| 缺失长度 | 单点 | 固定长度块 |
| 实际场景 | 随机噪声 | 设备故障、连续缺失 |
| 填补难度 | 较易 | 较难（连续缺失） |

## 注意事项

1. **块长度选择**：
   - 默认 50 适合大多数场景
   - 高频数据可适当增大块长度
   - 块长度不宜过大，否则可能导致实际缺失率偏低

2. **实际缺失率**：
   - 由于块数量向下取整，实际缺失率可能略低于目标缺失率
   - 例如：目标 5%，实际可能为 3.96%

3. **块分布**：
   - 块在注错区间内随机分布
   - 保证块与块之间不重叠、不相邻
   - 每列独立生成块缺失

## 测试验证

所有测试均已通过：

- ✅ 块生成算法正确性
- ✅ 块不重叠验证
- ✅ 块不相邻验证
- ✅ 块数量计算正确性
- ✅ MI_all.py 集成测试

## 相关文件

- [BM.py](file:///home/decadent/TSF-Imputation-Analysis/Missing_Value_Injection/for_sundial/BM.py) - 核心实现
- [MI_all.py](file:///home/decadent/TSF-Imputation-Analysis/Missing_Value_Injection/for_sundial/MI_all.py) - 统一入口
- [inject_range_utils.py](file:///home/decadent/TSF-Imputation-Analysis/Missing_Value_Injection/for_sundial/inject_range_utils.py) - 注错区间工具
- [test_bm.py](file:///home/decadent/TSF-Imputation-Analysis/test_bm.py) - 测试脚本

---

**生成时间**：2026-04-04
**作者**：AI Assistant
**版本**：1.0
