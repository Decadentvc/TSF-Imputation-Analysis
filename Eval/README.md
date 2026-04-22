# Eval 模块说明

`Eval/` 目录提供统一的时间序列预测评测流程，支持多模型在同一套数据切窗与指标体系下进行对比。当前包含两类入口：

1. **通用评估入口**：`Eval/run_eval.py`（`clean` / `single` / `batch` 三种模式）。
2. **批量任务入口（带跳过机制）**：`Eval/run_batch_eval.py`（自动跳过已生成结果，适合长任务续跑）。

评测主流程（数据加载、滚动窗口生成、指标计算、结果保存）在 `Eval/eval_pipeline.py`，模型差异通过 Adapter 层（`Eval/model_adapters.py`）隔离。

---

## 一、快速开始（可直接复制执行）

> 下列指令均在 `Sundial` conda 环境下验证。按终端类型任选一条进入工程目录并激活环境，随后追加具体脚本命令。

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

### 1. 干净数据评估（`run_eval.py clean`）

```bash
# Sundial
python Eval/run_eval.py clean --model sundial --dataset ETTh1 --term short

# Chronos-2
python Eval/run_eval.py clean --model chronos2 --dataset ETTh1 --term medium --device cuda:0 --torch_dtype bfloat16

# TimesFM 2.5
python Eval/run_eval.py clean --model timesfm2p5 --dataset ETTh1 --term long
```

### 2. 单文件缺失评估（`run_eval.py single`）

```bash
# 评估一个注空文件（先填补，再与 clean 真值对比）
python Eval/run_eval.py single --model chronos2 --eval_data_path data/datasets/Block_Missing/BM_010/ETTh1_BM_length50_010_short.csv --imputation_method linear
```

### 3. 批量缺失评估（`run_eval.py batch`）

```bash
# 指定缺失率和填补方法
python Eval/run_eval.py batch --model sundial --dataset ETTh1 --method BM --missing_ratios 0.10,0.20,0.30 --imputation_methods linear,mean,forward

# 使用默认缺失率（0.05~0.30）
python Eval/run_eval.py batch --model timesfm2p5 --dataset ETTh1 --method BM --imputation_methods linear,mean
```

### 4. 大规模批处理（`run_batch_eval.py`）

```bash
# 单数据集 + 多 term + 自动跳过已有结果
python Eval/run_batch_eval.py --model sundial --dataset ETTh1 --method BM --terms short,medium,long --imputation_methods linear,mean,forward --missing_ratios 0.10,0.20,0.30

# 扫描全部数据集 + 同时跑 clean 与 impute
python Eval/run_batch_eval.py --model chronos2 --include_clean --imputation_methods linear,mean --missing_ratios 10 20 30

# 强制重跑（忽略已存在结果）
python Eval/run_batch_eval.py --model timesfm2p5 --dataset ETTh1 --force
```

---

## 二、脚本详解

### A. `run_eval.py` — 通用评估入口

支持 3 种子命令：

| 子命令 | 用途 | 关键参数 |
| --- | --- | --- |
| `clean` | 使用干净数据评估（`eval == clean`） | `--dataset`, `--term` |
| `single` | 评估单个缺失文件（先填补再评测） | `--eval_data_path`, `--imputation_method` |
| `batch` | 批量评估缺失文件组合 | `--dataset`, `--method BM`, `--missing_ratios`, `--imputation_methods` |

核心行为：
- 若未显式传 `--prediction_length`，根据 `frequency + term` 自动计算：`short=1x`、`medium=10x`、`long=15x` 基础长度。
- 缺失数据评估时会先检查填补文件是否存在，不存在则自动调用 `Eval/impute_dataset.py` 生成。
- 评测结果会同时保存：
  - 最终指标 CSV：`results/<model>/clean|impute/...`
  - 窗口级预测中间文件：`data/Intermediate_Predictions/<model>/...`

### B. `run_batch_eval.py` — 带“跳过已存在结果”的批量入口

特点：
- 支持数据集自动扫描（`--dataset` 省略时遍历 `data/datasets/ori/*.csv`）。
- 默认开启“结果已存在则跳过”，可用 `--force` 覆盖重跑。
- 支持 `--include_clean` / `--clean_only`，可与缺失评测组合。
- 自动清理运行时缓存（`gc` 与 CUDA cache，best-effort）。

该脚本内部复用 `run_eval.py` 的 `evaluate_clean` 与 `run_single_evaluation`，评测逻辑保持一致。

### C. `impute_dataset.py` — 独立填补模块

提供缺失值填补与路径规范化能力，`run_eval.py` / `run_batch_eval.py` 在缺失评测前会自动调用。

支持填补方法：
- `zero`, `mean`, `forward`, `backward`, `linear`, `nearest`, `spline`, `seasonal`, `none`

说明：评估流程中不允许 `none` 作为缺失评测方法。

---

## 三、模型支持与扩展

### 当前 `run_eval.py` 支持模型

- `sundial`
- `chronos2`
- `timesfm2p5`
- `kairos23m`
- `kairos50m`
- `timesfm2p0`
- `visiontspp`

> 其中 `run_batch_eval.py` 当前限制为 `sundial` / `chronos2` / `timesfm2p5`。

<<<<<<< HEAD
### 模型切换方式

- 常规仅需切换 `--model`。
- 如需指定权重或 checkpoint，使用 `--model_name`。
- 通用参数（如 `--batch_size`、`--device`）对所有模型可传；不适用参数由适配层忽略或处理。
=======
- 只改命令行中的 `--model`，其余流程（数据加载、窗口生成、指标评估、结果保存）不需要改
- 如需指定不同权重，使用 `--model_name`
- 模型特有参数：
  - `sundial`：主要使用 `--num_samples`
  - `chronos2`：可用 `--predict_batches_jointly`、`--torch_dtype`
  - `timesfm2p5`：通常只需 `--batch_size`、`--device`（`--model_name` 可覆盖默认 checkpoint）
  - `kairos23m`：主要使用通用参数，默认 `--model_name mldi-lab/Kairos_23m`，通过远程仓库加载模型，不依赖本地 forecastor。
  - `kairos50m`：主要使用通用参数，默认 `--model_name mldi-lab/Kairos_50m`，通过远程仓库加载模型，不依赖本地 forecastor。
  - `timesfm2p0`：主要使用通用参数，默认 `--model_name google/timesfm-2.0-500m-pytorch`，已支持按 `prediction_length` 动态扩展 horizon（支持 long term）。
  - `visiontspp`：主要使用通用参数，默认 `--model_name Lefei/VisionTSpp`，通过 Hugging Face 下载模型，不依赖本地 forecastor。
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f

### 如何添加新模型

1. 在 `Eval/model_adapters.py` 新增 Adapter，实现统一 `predict(test_data_input)` 接口。
2. 在 `Eval/model_registry.py` 的 `build_model_adapter(...)` 注册模型名到 Adapter 的映射。
3. 在 `Eval/run_eval.py` 的 `--model` choices 中加入该模型。
4. （可选）在 `Eval/run_batch_eval.py` 的 `--model` choices 同步开放批量入口。

---

## 四、参数说明

### `run_eval.py` 通用参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | 模型名 | `sundial` |
| `--model_name` | 自定义权重名 / checkpoint | `None` |
| `--base_data_dir` | 数据根目录 | `data/datasets` |
| `--properties_path` | 数据属性文件 | `data/datasets/dataset_properties.json` |
| `--output_dir` | 结果目录（可覆盖默认） | 按模式自动到 `results/<model>/...` |
| `--prediction_length` | 预测长度；不传则自动计算 | `None` |
| `--num_samples` | 采样数（部分模型有效） | `100` |
| `--batch_size` | 推理 batch 大小 | `32` |
| `--device` | 运行设备 | `cpu` |
| `--intermediate_dir` | 中间预测根目录 | `data/Intermediate_Predictions` |
| `--predict_batches_jointly` | Chronos-2 联合批预测开关 | 关闭 |
| `--torch_dtype` | Chronos-2 dtype（`bfloat16/float16/float32`） | `None` |

### `run_eval.py` 子命令补充参数

- `single`：`--eval_data_path`（必填）、`--clean_data_path`（可选）、`--term`（可选）、`--imputation_method`（必填）
- `batch`：`--dataset`（必填）、`--method BM`（当前仅 BM）、`--missing_ratios`、`--imputation_methods`、`--block_length`
- `clean`：`--dataset`（必填）、`--term`（默认 `short`）

### `run_batch_eval.py` 特色参数

- `--terms`：支持空格和逗号混输，如 `short,medium long`
- `--missing_ratios`：支持 `0.1` / `10` / `010` 三类写法
- `--include_clean`：在批量缺失评测时附带 clean 评测
- `--clean_only`：只跑 clean
- `--force`：即使结果已存在也重跑

---

## 五、目录与命名规范

### 输入数据目录（模型共享）

- `data/datasets/ori/`
- `data/datasets/Block_Missing/`（新目录优先）
- `data/datasets/BM/`（旧目录兼容）
- `data/datasets/Imputed/`
- `data/datasets/dataset_properties.json`

### 缺失数据文件命名（BM）

`run_eval.py` 在 BM 场景支持以下命名：

- 带 block 长度：`{dataset}_BM_length{len}_{ratio}_{term}.csv`
- 不带 block 长度：`{dataset}_BM_{ratio}_{term}.csv`

其中 `ratio` 使用三位百分比，如 `010` 表示 10%。

### 输出结构

```
results/<model>/
├── clean/
│   └── {dataset}_clean_{term}_results.csv
└── impute/
    └── {impute_method}_{eval_name}_results.csv

data/Intermediate_Predictions/<model>/
└── {eval_name}_prediction/
    ├── {eval_name}_prediction_{window_idx}.csv
    └── {impute_method}/
        └── {eval_name}_prediction_{window_idx}.csv
```

填补数据缓存（自动生成）：

```
data/datasets/Imputed/{METHOD}/{METHOD}_{RATIO}/{dataset}_{METHOD}_{RATIO}_{term}_{impute_method}.csv
```

---

## 六、评测指标与窗口策略

指标由 `Eval/eval_pipeline.py` 统一计算，当前包含：

- `MSE[mean]`, `MSE[0.5]`
- `MAE[0.5]`, `MASE[0.5]`, `MAPE[0.5]`, `sMAPE[0.5]`
- `MSIS`, `RMSE[mean]`, `NRMSE[mean]`, `ND[0.5]`
- `mean_weighted_sum_quantile_loss`

窗口策略：
- 评估窗口数基于 `TEST_SPLIT=0.6` 与 `MAX_WINDOW=20` 计算。
- 常规数据集采用滚动窗口；`m4` 数据集固定 `windows=1`。

---

## 七、注意事项

<<<<<<< HEAD
- `single` / `batch` 缺失评测必须使用有效填补方法，`none` 不允许。
- `term` 需匹配数据集属性：`short` 或 `short,medium,long`（由 `dataset_properties.json` 的 `term` 字段决定）。
- 自动预测长度依赖 `dataset_properties.json` 的 `frequency` 配置。
- `run_batch_eval.py` 默认跳过已有结果，长任务续跑时无需手工筛选；全量重跑请加 `--force`。
=======
## 输出文件命名

### clean

- `results/<model>/clean/{dataset}_clean_{term}_results.csv`

### impute

- `results/<model>/impute/{impute_method}_{eval_name}_{term}_results.csv`

### 中间预测

- clean（无填补）：`data/Intermediate_Predictions/<model>/{eval_name}_prediction/{eval_name}_prediction_{window_idx}.csv`
- impute（有填补）：`data/Intermediate_Predictions/<model>/{eval_name}_prediction/{imputation_method}/{eval_name}_prediction_{window_idx}.csv`

## 依赖

按模型场景安装依赖：

- 通用：`numpy`, `pandas`, `torch`, `gluonts`, `transformers`, `tqdm`
- `chronos2`：`chronos-forecasting>=2.1`
- `timesfm2p5`：`timesfm`（需包含 `timesfm_2p5_torch`）
- `kairos23m` / `kairos50m`：需要可用 `tsfm`（包含 `tsfm.model.kairos`）。
- `timesfm2p0`：需要支持 TimesFM 的 `transformers` 版本（包含 `TimesFmModelForPrediction`）。
- `visiontspp`：需要 `visionts` 与 `huggingface_hub`。

如报 `ModuleNotFoundError`，请先在当前环境补齐依赖。

## 常见注意事项

- `single` 模式缺失数据评估必须提供 `--imputation_method`，且不能是 `none`
- `batch` 模式默认不包含 `none` 填补方法
- `term` 必须与数据任务匹配：`short` / `medium` / `long`
- `dataset_properties.json` 必须包含对应数据集的 `frequency` 与 `term` 信息
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f
