# 通用模型评估说明（新版本）

本文件对应新入口 `Eval/run_eval.py`，用于统一评估不同时序模型（当前支持 `sundial` 与 `chronos2`）。

旧版 `Eval/run_sundial.py` / `Eval/eval_sundial.py` 仍保留，本说明不替换原有 README。

## 1. 目标

- 使用同一套命令行接口评估不同模型
- 抽离模型无关流程（数据加载、窗口构造、指标计算、结果保存）
- 模型相关逻辑放到适配层，后续可继续扩展更多模型

## 2. 新文件结构

- `Eval/run_eval.py`：新 CLI 入口（single/batch/clean）
- `Eval/eval_pipeline.py`：通用评估管线
- `Eval/model_registry.py`：模型注册与构建
- `Eval/model_adapters.py`：模型适配器实现（Sundial/Chronos-2）

## 3. 目录约定

### 3.1 数据目录（模型共享）

- `data/datasets/ori/`
- `data/datasets/Block_Missing/`
- `data/datasets/Imputed/`
- `data/datasets/dataset_properties.json`

说明：上述目录不按模型拆分，所有模型共用。

### 3.2 中间预测（按模型区分）

- `data/Intermediate_Predictions/<model>/...`

例如：
- `data/Intermediate_Predictions/sundial/...`
- `data/Intermediate_Predictions/chronos2/...`

### 3.3 最终评测结果（按模型区分）

- `results/<model>/clean/...`
- `results/<model>/impute/...`

说明：
- `clean`：使用干净数据评估
- `impute`：对缺失数据先填补，再评估
- 新逻辑下不保留 missing 场景中的 `none`（不填补）评估分支

## 4. 运行模式

`run_eval.py` 支持 3 种模式：

- `single`：评估单个缺失数据文件（必须指定填补方法）
- `batch`：按数据集、缺失率、term、填补方法批量评估
- `clean`：在干净数据上评估

## 5. 参数说明

### 5.1 通用参数

- `--model`：模型类型（`sundial` / `chronos2`）
- `--model_name`：具体模型权重名称（可选）
- `--base_data_dir`：数据根目录（默认 `data/datasets`）
- `--properties_path`：属性文件（默认 `data/datasets/dataset_properties.json`）
- `--output_dir`：结果输出目录（可选，默认按模型自动分流）
- `--prediction_length`：预测长度（可选，不传则按 frequency + term 自动计算）
- `--batch_size`：推理批次大小
- `--device`：运行设备（如 `cpu`、`cuda:0`）
- `--num_samples`：采样数（主要用于 Sundial）
- `--intermediate_dir`：中间结果根目录（默认 `data/Intermediate_Predictions`）

### 5.2 Chronos-2 相关参数

- `--predict_batches_jointly`：是否 joint 预测批次
- `--torch_dtype`：`bfloat16` / `float16` / `float32`

说明：
- 不同模型参数通过适配器处理。
- 某模型不使用的参数会被忽略，不影响运行。

## 6. 使用示例

### 6.1 Clean 评估

```bash
python Eval/run_eval.py clean \
  --model sundial \
  --dataset ETTh1 \
  --term short
```

### 6.2 Single 评估（缺失数据 + 填补）

```bash
python Eval/run_eval.py single \
  --model chronos2 \
  --eval_data_path "data/datasets/Block_Missing/BM_010/ETTh1_BM_length50_010_short.csv" \
  --imputation_method linear
```

### 6.3 Batch 评估

```bash
python Eval/run_eval.py batch \
  --model sundial \
  --dataset ETTh1 \
  --method BM \
  --missing_ratios "0.10,0.20,0.30" \
  --imputation_methods "linear,mean,forward"
```

## 7. 输出文件命名

### 7.1 clean

- `results/<model>/clean/{dataset}_clean_{term}_results.csv`

### 7.2 impute

- `results/<model>/impute/{impute_method}_{eval_name}_{term}_results.csv`

### 7.3 中间预测

- 无填补（clean）：
  - `data/Intermediate_Predictions/<model>/{eval_name}_prediction/{eval_name}_prediction_{window_idx}.csv`
- 有填补（impute）：
  - `data/Intermediate_Predictions/<model>/{eval_name}_prediction/{imputation_method}/{eval_name}_prediction_{window_idx}.csv`

## 8. 依赖

至少需要以下依赖（按模型场景安装）：

- 通用：`numpy`, `pandas`, `torch`, `gluonts`, `transformers`, `tqdm`
- Chronos-2：`chronos-forecasting>=2.1`

如果运行时报 `ModuleNotFoundError`，请先在当前环境补齐依赖。

## 9. 常见注意事项

- `single` 模式下，缺失数据评估必须提供 `--imputation_method`，且不能是 `none`
- `batch` 模式默认不包含 `none` 填补方法
- `term` 必须与数据/任务匹配：`short` / `medium` / `long`
- `dataset_properties.json` 必须包含对应数据集的 `frequency` 与 `term` 信息
