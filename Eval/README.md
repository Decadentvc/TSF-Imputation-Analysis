# 通用模型评估说明（run_eval）

本文件对应新入口 `Eval/run_eval.py`，用于统一评估不同时序模型。旧版 `Eval/run_sundial.py` / `Eval/eval_sundial.py` 仍保留。

## 快速运行指令（放在最前）

```bash
python Eval/run_eval.py clean --model sundial --dataset ETTh1 --term short
python Eval/run_eval.py clean --model chronos2 --dataset ETTh1 --term short
python Eval/run_eval.py clean --model timesfm2p5 --dataset ETTh1 --term short
python Eval/run_eval.py single --model chronos2 --eval_data_path "data/datasets/Block_Missing/BM_010/ETTh1_BM_length50_010_short.csv" --imputation_method linear
python Eval/run_eval.py batch --model sundial --dataset ETTh1 --method BM --missing_ratios "0.10,0.20,0.30" --imputation_methods "linear,mean,forward"
python Eval/run_eval.py batch --model timesfm2p5 --dataset ETTh1 --method BM --imputation_methods "linear,mean"
```

### 指令说明

- `clean`：使用干净数据评估，`eval_data_path == clean_data_path`
- `single`：评估单个缺失数据文件，必须传 `--imputation_method`（不允许 `none`）
- `batch`：按缺失率、term、填补方法批量评估
- 若未传 `--prediction_length`，会按 `frequency + term` 自动计算

## 可用模型与切换方式（放在指令说明后）

当前 `--model` 可选：
- `sundial`
- `chronos2`
- `timesfm2p5`

### 使用不同模型时需要改什么

- 只改命令行中的 `--model`，其余流程（数据加载、窗口生成、指标评估、结果保存）不需要改
- 如需指定不同权重，使用 `--model_name`
- 模型特有参数：
  - `sundial`：主要使用 `--num_samples`
  - `chronos2`：可用 `--predict_batches_jointly`、`--torch_dtype`
  - `timesfm2p5`：通常只需 `--batch_size`、`--device`（`--model_name` 可覆盖默认 checkpoint）

### 如何添加新模型

按以下固定步骤扩展：

1. 在 `Eval/model_adapters.py` 新增 Adapter（实现统一接口 `predict(test_data_input)`，返回 `SampleForecast` 或 `QuantileForecast`）
2. 在 `Eval/model_registry.py` 的 `build_model_adapter(...)` 注册模型字符串到 Adapter 的映射
3. 在 `Eval/run_eval.py` 的 `--model` 参数 `choices` 中加入新模型名
4. （可选）在本 README 的“快速运行指令”和“参数说明”补充该模型示例

说明：`Eval/eval_pipeline.py` 是模型无关评估管线，正常情况下不需要改。

## 目标

- 使用同一套 CLI 评估不同模型
- 抽离模型无关流程（数据加载、窗口构造、指标计算、结果保存）
- 模型相关逻辑集中在适配层，便于持续扩展

## 文件结构

- `Eval/run_eval.py`：CLI 入口（`single` / `batch` / `clean`）
- `Eval/eval_pipeline.py`：通用评估管线
- `Eval/model_registry.py`：模型注册与构建
- `Eval/model_adapters.py`：模型适配器实现

## 目录约定

### 数据目录（模型共享）

- `data/datasets/ori/`
- `data/datasets/Block_Missing/`
- `data/datasets/Imputed/`
- `data/datasets/dataset_properties.json`

说明：数据目录不按模型拆分，所有模型共用。

### 中间预测（按模型区分）

- `data/Intermediate_Predictions/<model>/...`

例如：
- `data/Intermediate_Predictions/sundial/...`
- `data/Intermediate_Predictions/chronos2/...`
- `data/Intermediate_Predictions/timesfm2p5/...`

### 最终评测结果（按模型区分）

- `results/<model>/clean/...`
- `results/<model>/impute/...`

说明：
- `clean`：干净数据评估
- `impute`：缺失数据先填补再评估
- 当前逻辑不保留 missing 场景的 `none`（不填补）分支

## 运行模式

`run_eval.py` 支持 3 种模式：

- `single`：单个缺失文件评估（必须指定填补方法）
- `batch`：按数据集、缺失率、term、填补方法批量评估
- `clean`：干净数据评估

## 参数说明

### 通用参数

- `--model`：模型类型（`sundial` / `chronos2` / `timesfm2p5`）
- `--model_name`：模型权重名称（可选）
- `--base_data_dir`：数据根目录（默认 `data/datasets`）
- `--properties_path`：属性文件（默认 `data/datasets/dataset_properties.json`）
- `--output_dir`：结果输出目录（可选；默认按模型自动分流）
- `--prediction_length`：预测长度（可选；不传则自动计算）
- `--batch_size`：推理批次大小
- `--device`：运行设备（如 `cpu`、`cuda:0`）
- `--num_samples`：采样数（主要用于 `sundial`）
- `--intermediate_dir`：中间结果根目录（默认 `data/Intermediate_Predictions`）

### Chronos-2 相关参数

- `--predict_batches_jointly`：是否 joint 预测批次
- `--torch_dtype`：`bfloat16` / `float16` / `float32`

说明：
- 不同模型参数由适配器处理
- 某模型不使用的参数会被忽略，不影响运行

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

如报 `ModuleNotFoundError`，请先在当前环境补齐依赖。

## 常见注意事项

- `single` 模式缺失数据评估必须提供 `--imputation_method`，且不能是 `none`
- `batch` 模式默认不包含 `none` 填补方法
- `term` 必须与数据任务匹配：`short` / `medium` / `long`
- `dataset_properties.json` 必须包含对应数据集的 `frequency` 与 `term` 信息
