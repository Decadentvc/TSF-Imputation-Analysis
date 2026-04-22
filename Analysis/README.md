# Analysis 模块说明

`Analysis/` 目录提供窗口级别的时间序列特征分析工具，对以下三类窗口批量计算 6 个基于 STL 分解的特征指标：

1. **模型预测窗口**：`data/Intermediate_Predictions/<model>/...` 下每个模型输出的预测片段（含干净与填补两类来源）。
2. **历史窗口**：预测窗口紧邻的前 `max_context` 段历史（可来自干净数据或填补数据）。
3. **干净预测真实值窗口**：`data/datasets/ori/<dataset>.csv` 中与预测区间对齐的真值切片，用作与预测窗口的 Ground-Truth 比较。

---

## 一、快速开始（可直接复制执行）

> 下列指令均在 `Sundial` conda 环境下验证。按所用终端类型任选一条进入工程目录并激活环境，随后追加具体脚本命令。

- Git Bash：
  ```bash
  cd /d/Project/TSF-Imputation-Analysis && source /d/anaconda3/Scripts/activate Sundial
  ```
- Windows CMD：
  ```cmd
  cd /d d:\Project\TSF-Imputation-Analysis && call D:\anaconda3\Scripts\activate.bat Sundial
  ```
- PowerShell：
  ```powershell
  cd d:\Project\TSF-Imputation-Analysis; & D:\anaconda3\shell\condabin\conda-hook.ps1; conda activate Sundial
  ```

### 1. 模型预测/历史窗口批量分析（`run_batch_analysis.py`）

```bash
# 默认：对 sundial 的所有可用 dataset × term × 填补方法跑预测 + 历史窗口分析
python Analysis/run_batch_analysis.py --model sundial

# 多模型 + 指定 term
python Analysis/run_batch_analysis.py --model sundial chronos2 --terms short,medium,long

# 仅跑干净数据的预测窗口
python Analysis/run_batch_analysis.py --model sundial --clean_only --prediction_only

# 指定单个数据集 + 指定填补方法 + 强制覆盖
python Analysis/run_batch_analysis.py --model timesfm2p5 --dataset ETTh1 --imputation_methods linear,mean --force
```

### 2. 干净预测真实值窗口分析（`run_clean_prediction_window_analysis.py`）

```bash
# 默认：扫描 data/datasets/ori/ 下全部数据集，term 按 dataset_properties.json 自动选
python Analysis/run_clean_prediction_window_analysis.py

# 指定若干数据集 + 全部 term
python Analysis/run_clean_prediction_window_analysis.py --dataset ETTh1 ETTh2 ETTm1 --terms short,medium,long

# 仅分析 short term，并强制覆盖既有结果
python Analysis/run_clean_prediction_window_analysis.py --terms short --force
```

### 3. 单窗口组合交互式分析（`window_analysis.py`）

```bash
# 预测窗口（干净）
python -m Analysis.window_analysis --model sundial prediction --prediction_dir data/Intermediate_Predictions/sundial/ETTh1_clean_long_prediction

# 预测窗口（填补方法子目录）
python -m Analysis.window_analysis --model sundial prediction --prediction_dir data/Intermediate_Predictions/sundial/ETTh1_BM_length50_010_long_prediction/linear

# 干净历史窗口
python -m Analysis.window_analysis --model sundial history --dataset ETTh1 --term long
```

---

## 二、脚本详解

### A. `run_batch_analysis.py` — 模型预测 + 历史窗口批量分析

覆盖范围：
- **预测窗口**：遍历 `data/Intermediate_Predictions/<model>/` 下所有符合命名的目录（`*_clean_*_prediction/` 与 `*_{method}_length*_*_*_prediction/<impute_method>/`），对每个窗口 CSV 计算 6 个指标并聚合。
- **历史窗口**：按 `inject_range_utils` 从干净数据或填补数据中截取每个预测窗口前 `max_context` 段历史做分析。

核心参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | 模型名称，可多选，必填，取值：`sundial` / `chronos2` / `timesfm2p5` | — |
| `--dataset` | 单个数据集过滤；省略则跑该模型下全部发现的数据集 | 全部 |
| `--terms` | 逗号或空格分隔的 term，例如 `short,medium,long` | 全部 |
| `--imputation_methods` | 要分析的填补方法 | `linear,mean,forward,backward` |
| `--intermediate_dir` | 模型预测结果根目录 | `data/Intermediate_Predictions` |
| `--output_dir` | 结果输出根目录 | `results_analysis` |
| `--properties_path` | 数据集属性 JSON | `data/datasets/dataset_properties.json` |
| `--prediction_only` | 仅跑预测窗口分析 | 否 |
| `--history_only` | 仅跑历史窗口分析 | 否 |
| `--clean_only` | 跳过填补方法，只跑 clean 预测窗口 | 否 |
| `--force` | 覆盖已有输出文件 | 否 |

输出结构：
```
results_analysis/<model>/
├── prediction/
│   ├── {dataset}_clean_{term}_prediction.csv
│   ├── {dataset}_clean_{term}_prediction_summary.json
│   ├── {dataset}_{method}_{ratio}_{term}_{impute}_prediction.csv
│   └── ...
├── history/
│   ├── {dataset}_clean_{term}_history.csv
│   ├── {dataset}_{method}_{ratio}_{term}_{impute}_history.csv
│   └── ...
├── overall_prediction_summary.json
└── overall_history_summary.json
```

### B. `run_clean_prediction_window_analysis.py` — 干净预测真实值窗口分析

作用：在 `data/datasets/ori/<dataset>.csv` 尾部按 `inject_range_utils.get_injection_range` 给出的 `prediction_length × windows` 切出真值窗口（模型输出窗口对应的 Ground-Truth），对每个窗口计算 6 个 STL 指标。该切分与模型 `max_context` 无关，因此脚本不需要 `--model`。

核心参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset` | 一个或多个数据集名；省略则扫描 `data/datasets/ori/` 下全部 CSV | 全部 |
| `--terms` | 一个或多个 term；省略则按 `dataset_properties.json` 的 `term` 字段自动选 | 自动 |
| `--data_path` | 数据集根目录 | `data/datasets` |
| `--properties_path` | 数据集属性 JSON | `data/datasets/dataset_properties.json` |
| `--output_dir` | 结果输出目录 | `results_analysis/clean_prediction_windows` |
| `--force` | 覆盖已有输出文件 | 否 |

输出结构：
```
results_analysis/clean_prediction_windows/
├── {dataset}_clean_{term}_prediction_gt.csv          # 每窗口指标明细
├── {dataset}_clean_{term}_prediction_gt_summary.json # 元信息 + 均值/std/min/max
└── overall_clean_prediction_summary.json             # 全局 + per-term 聚合
```

### C. `window_analysis.py` — 单次交互式分析

通过 `mode` 子命令选择分析对象，适用于对单个目录或单个数据集做即席分析。

| 子命令 | 关键参数 | 用途 |
| --- | --- | --- |
| `prediction` | `--prediction_dir`, `--output` | 分析单个预测窗口目录 |
| `history` | `--dataset`, `--term`, `--output` | 分析单个数据集的干净历史窗口 |

所有子命令共享全局参数：`--model`（必填）、`--data_path`、`--properties_path`、`--model_properties_path`、`--output_root`。

---

## 三、目录命名规范

预测结果目录必须匹配以下两种模式之一，否则会被跳过：

- **干净数据**：`{dataset}_clean_{term}_prediction/`
- **填补数据**：`{dataset}_{method}_length{len}_{ratio}_{term}_prediction/{impute_method}/`
  - `method` ∈ `MCAR | BM | TM | TVMR`
  - `ratio` 为三位数百分比（`010` 表示 10%）
  - `impute_method` ∈ `zero | mean | forward | backward | linear | nearest | spline | seasonal`

---

## 四、指标说明

所有指标均基于各列独立 STL 分解 `(trend, seasonal, residual)`，再对同一窗口内的所有数值列取平均。

1. **趋势强度 `trend_strength`**：`F_TS = max(0, 1 - Var(R) / Var(T + R))`，范围 [0, 1]。
2. **趋势线性度 `trend_linearity`**：将趋势分量拟合 `T_t = β0 + β1·P1(t) + β2·P2(t)`，取 `β1`。>0 表上升。
3. **季节强度 `seasonal_strength`**：`F_SS = max(0, 1 - Var(R) / Var(S + R))`，范围 [0, 1]。
4. **季节相关性 `seasonal_correlation`**：截取整数个周期，所有季节段两两皮尔逊相关取均值，范围 [-1, 1]。
5. **残差一阶自相关 `residual_autocorr_lag1`**：`F_RA = E[(R_t-R̄)(R_{t-1}-R̄)] / Var(R)`，范围 [-1, 1]。
6. **谱熵 `spectral_entropy`**：`Σ log(PSD)`，衡量频谱复杂度，值越大频率越分散。

实现见 [Analysis/metrics.py](metrics.py)。

---

## 五、注意事项

- 运行前确保 `data/datasets/dataset_properties.json` 包含目标数据集的 `period`、`frequency` 和 `term` 配置。
- STL 分解要求每列有效样本数 `>= 2 * period`，不足会跳过该列；若窗口内所有列都不满足，该窗口整体失败，终端显示 `[SKIP-DATA]` 或 `Insufficient data`。
- `run_batch_analysis.py` 的模型 `max_context` 取自 [Eval/model_properties.json](../Eval/model_properties.json)，历史窗口的长度即由此决定。
- `run_clean_prediction_window_analysis.py` 的窗口数仅取决于数据集总长、`prediction_length` 与 `MAX_WINDOW=20`（见 [inject_range_utils.py](../tools/Missing_Value_Injection/inject_range_utils.py)），不同模型的结果相同，因此无需区分模型目录。
- `--force` 会覆盖已有文件；默认行为是跳过已存在结果。
