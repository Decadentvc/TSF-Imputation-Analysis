# Missing Value Injection（BM-only）

当前目录仅保留 **BM（Block Missing，块缺失）** 注入能力。

## 目录说明

```text
tools/Missing_Value_Injection/
├── BM.py                  # 单数据集/单命令入口（支持自动 term、多缺失率）
├── batch_bm_injection.py  # 批量遍历 datasets/ori 的全部数据集
├── inject_range_utils.py  # 注错区间计算工具
└── README.md
```

## 环境准备

```bash
source /home/decadent/miniconda3/etc/profile.d/conda.sh
conda activate TSFIA
cd /home/decadent/TSF-Imputation-Analysis
```

---

## 1) BM.py：常用注空入口

查看帮助：

```bash
python tools/Missing_Value_Injection/BM.py --help
```

### 示例 A：单个缺失率，自动 term

```bash
python tools/Missing_Value_Injection/BM.py \
  --dataset ETTh1 \
  --missing_ratio 0.05
```

> 默认 `--data_path` 与 `--output_dir` 已指向 `data/datasets`，一般无需手动填写。

### 示例 B：多个缺失率

```bash
python tools/Missing_Value_Injection/BM.py \
  --dataset ETTh1 \
  --missing_ratio "0.05,0.1,0.15" \
  --block_length 50
```

### 示例 C：只跑指定 term（关闭自动 term）

```bash
python tools/Missing_Value_Injection/BM.py \
  --dataset ETTh1 \
  --missing_ratio 0.1 \
  --term short \
  --no_auto_term
```

### 示例 D：修改 max_context（默认 8192）

```bash
python tools/Missing_Value_Injection/BM.py \
  --dataset exchange_rate \
  --missing_ratio 0.1 \
  --max_context 4096
```

### 示例 E：显式指定数据目录

```bash
python tools/Missing_Value_Injection/BM.py \
  --dataset Finland_Traffic_15T \
  --missing_ratio 0.1 \
  --term short \
  --no_auto_term \
  --data_path data/datasets
```

### 输出路径

输出文件格式：

```text
{output_dir}/BM/BM_{ratio}/{dataset}_BM_length{block_length}_{ratio}_{term}.csv
```

例如：

```text
data/datasets/BM/BM_010/ETTh1_BM_length50_010_short.csv
```

---

## 2) batch_bm_injection.py：批量全数据集注空

该脚本会遍历 `datasets/ori/*.csv`，并依据 `dataset_properties.json` 的 term 配置执行注空。

查看帮助：

```bash
python tools/Missing_Value_Injection/batch_bm_injection.py --help
```

### 示例 A：默认配置批量执行

```bash
python tools/Missing_Value_Injection/batch_bm_injection.py
```

### 示例 B：自定义缺失率、块长度、max_context

```bash
python tools/Missing_Value_Injection/batch_bm_injection.py \
  --missing_ratios "0.05,0.1" \
  --block_length 100 \
  --max_context 8192 \
  --seed 123
```

---

## 3) inject_range_utils.py：单独查看注错区间

查看帮助：

```bash
python tools/Missing_Value_Injection/inject_range_utils.py --help
```

示例：

```bash
python tools/Missing_Value_Injection/inject_range_utils.py \
  --dataset ETTh1 \
  --term long \
  --max_context 8192
```

---

## 参数补充

- `missing_ratio` / `missing_ratios` 支持：
  - `0.05`
  - `0.05,0.1,0.15`
  - `[0.05,0.1,0.15]`
- `max_context` 默认值为 `8192`。
- BM 注入只作用于非时间列与非标识列（`date/time/timestamp/item_id` 会跳过）。
