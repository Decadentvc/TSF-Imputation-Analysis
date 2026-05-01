# TSF-Imputation-Analysis

本仓库用于研究块状缺失（Block Missing, BM）场景下，不同缺失填补方法对时间序列基础预测模型（Time Series Foundation Models, TSFM）下游预测性能的影响。当前流程覆盖数据缺失注入、缺失数据填补、预测模型评估、窗口级结构特征分析和结果可视化。

核心实验链路是：

```text
原始时间序列 -> 块状缺失注入 -> 缺失填补 -> TSFM 预测 -> 指标评估与窗口特征分析 -> 可视化
```

## 仓库功能

- 生成块状缺失数据：在指定预测任务的可评估区间内注入固定长度的连续缺失块。
- 对缺失数据做填补：支持均值、前向、后向、线性插值，以及更多已注册的结构化/模型化填补方法。
- 统一评估多个预测模型：通过同一套 `Eval/run_eval.py` 和 `Eval/run_batch_eval.py` 接入不同 TSFM。
- 保存逐窗口预测结果：便于后续分析预测窗口与历史窗口之间的结构差异。
- 分析时间序列结构特征：基于 STL 等方法计算趋势、季节性、残差自相关、频谱复杂度等窗口级指标。
- 绘制实验散点图：比较结构扰动与预测误差退化之间的关系。

## 当前已有结果的数据集

仓库中原始数据集不止这些。README 只列出当前已经生成 BM 缺失数据并已有填补预测结果的数据集，位置主要在 `data/datasets/BM/BM_010`、`BM_020`、`BM_030` 和 `results/<model>/impute`。

已覆盖的数据集共 25 个：

`Australia_Solar_H`、`azure2019_U_5T`、`Coastal_T_S_15T`、`Coastal_T_S_20T`、`Coastal_T_S_5T`、`Coastal_T_S_H`、`current_velocity_15T`、`current_velocity_20T`、`current_velocity_5T`、`current_velocity_H`、`electricity`、`ETTh1`、`ETTh2`、`EWELD_Load_15T`、`exchange_rate`、`Finland_Traffic_15T`、`national_illness`、`NE_China_Wind_H`、`OpenElectricity_NEM_5T`、`Port_Activity_D`、`Supply_Chain_Customer_D`、`Supply_Chain_Location_D`、`traffic`、`Water_Quality_Darwin_15T`、`weather`。

其中 `chronos2` 和 `sundial` 的 impute 结果覆盖上述 25 个数据集；`kairos23m`、`kairos50m`、`timesfm2p0`、`timesfm2p5`、`visiontspp` 的 impute 结果当前覆盖其中 24 个，缺少 `national_illness`。clean 基线结果中还包含 `ETTm1`、`ETTm2` 等数据集，但它们当前不属于已生成 BM 缺失并完成填补预测结果的主实验集合。

## 缺失注入方式

当前实验使用 BM（Block Missing，块状缺失）注入方式，代码位于 `tools/Missing_Value_Injection/BM.py`。

- 缺失率：`10%`、`20%`、`30%`，对应目录 `BM_010`、`BM_020`、`BM_030`。
- 缺失块长度：当前已有数据文件使用 `block_length=50`。
- 注入模式：默认 `stratified`，会参考多个上下文长度区间（默认 `512,2048,2880,4096,8192`）尽量平衡不同模型上下文中的缺失率。
- 作用列：跳过 `date/time/timestamp/item_id` 等时间或标识列，仅对数值列注入缺失。
- 输出命名：`data/datasets/BM/BM_{ratio}/{dataset}_BM_length50_{ratio}_{term}.csv`。

示例：

```bash
python tools/Missing_Value_Injection/batch_bm_injection.py --missing_ratios 0.1,0.2,0.3 --block_length 50 --mode stratified
```

## 实现的填补算法

填补算法集中注册在 `Imputation/imputation_methods.py`，主评估流程通过 `Eval/impute_dataset.py` 调用。

当前已有预测结果主要使用 4 种填补方法：

- `mean`：按列均值填补。
- `forward`：前向填补。
- `backward`：后向填补。
- `linear`：线性插值。

代码中还实现并注册了以下方法，可用于继续扩展实验：

- `none`：不填补，保留 NaN。当前缺失数据评估流程不允许 `none` 作为正式 impute 分支。
- `zero`：零值填补。
- `nearest`：最近邻插值。
- `polynomial`：多项式插值。
- `spline`：样条插值。
- `seasonal`：季节分解填补，失败时回退到线性插值。
- `kalman_struct`：局部线性趋势状态空间模型 + Kalman smoother。
- `kalman_arima`：AR(p) 状态空间近似 + Kalman smoother。
- `stl_kalman`：STL/季节 profile 分解后对残差做 Kalman 填补。
- `gp_rbf`：一维时间索引上的 RBF Gaussian Process 填补。
- `saits`：基于 PyPOTS SAITS 的深度学习填补方法，需要额外安装 `pypots`。

## 实现的预测算法

预测模型统一通过 `Eval/model_adapters.py` 和 `Eval/model_registry.py` 适配，评估入口支持以下模型：

- `sundial`：默认权重 `thuml/sundial-base-128m`。
- `chronos2`：默认权重 `amazon/chronos-2`。
- `timesfm2p5`：默认权重 `google/timesfm-2.5-200m-pytorch`。
- `kairos23m`：默认权重 `mldi-lab/Kairos_23m`。
- `kairos50m`：默认权重 `mldi-lab/Kairos_50m`。
- `timesfm2p0`：默认权重 `google/timesfm-2.0-500m-pytorch`。
- `visiontspp`：默认权重 `Lefei/VisionTSpp`。

输出指标保存到 `results/<model>/clean` 和 `results/<model>/impute`。当前结果数量概览：

| 模型 | clean 结果数 | impute 结果数 | impute 方法 |
| --- | ---: | ---: | --- |
| `chronos2` | 79 | 444 | `mean`, `forward`, `backward`, `linear` |
| `sundial` | 79 | 444 | `mean`, `forward`, `backward`, `linear` |
| `timesfm2p5` | 79 | 288 | `mean`, `forward`, `backward`, `linear` |
| `kairos23m` | 26 | 288 | `mean`, `forward`, `backward`, `linear` |
| `kairos50m` | 45 | 336 | `mean`, `forward`, `backward`, `linear` |
| `timesfm2p0` | 26 | 288 | `mean`, `forward`, `backward`, `linear` |
| `visiontspp` | 26 | 288 | `mean`, `forward`, `backward`, `linear` |

## 目录结构

```text
.
├── Analysis/
├── data/
├── draw/
├── Eval/
├── Imputation/
├── results/
├── results_analysis/
├── tools/
├── 文档资料/
├── TSFIA.yml
└── README.md
```

### `Analysis/`

窗口级分析模块。

- `metrics.py`：实现趋势强度、趋势线性度、季节强度、季节相关性、残差一阶自相关、频谱熵等特征指标。
- `window_analysis.py`：对预测窗口、干净历史窗口、填补历史窗口做单次分析。
- `run_batch_analysis.py`：按模型、term、填补方法批量分析预测窗口与历史窗口。
- `run_clean_prediction_window_analysis.py`：生成 clean 预测窗口的对照特征。
- `imputed_evaluation.py`：评估填补数据相对原始数据的误差。

### `data/`

数据与中间预测结果目录。

- `data/datasets/ori/`：原始 CSV 数据集。
- `data/datasets/BM/BM_010|BM_020|BM_030/`：已注入 BM 缺失的数据。
- `data/datasets/Imputed/BM/BM_010|BM_020|BM_030/`：缺失数据经过不同填补方法处理后的 CSV。
- `data/datasets/dataset_properties.json`：数据集频率、变量数、领域、预测周期类型、周期长度等元信息。
- `data/Intermediate_Predictions/<model>/`：每个模型逐窗口预测结果，impute 场景下继续按填补方法分子目录。

### `draw/`

可视化脚本与输出图。

- `visualized_results_by_mode.py`：按单个填补方法汇总所有数据集绘图。
- `visualized_results_by_dataset.py`：按数据集绘图，在同一图中比较多个填补方法。
- `outputs_by_model/`：按模型和填补方法组织的散点图与 CSV。
- `outputs_by_dataset/`：按模型和数据集组织的散点图与 CSV。

### `Eval/`

预测评估主流程。

- `run_eval.py`：统一 CLI 入口，支持 `clean`、`single`、`batch` 三种模式。
- `run_batch_eval.py`：批量评估入口，支持跳过已有结果、强制重跑、只跑 clean、同时跑 clean 和 impute。
- `eval_pipeline.py`：模型无关的评估管线，包括窗口构造、预测长度计算、指标计算和结果保存。
- `model_adapters.py`：各预测模型的适配器。
- `model_registry.py`：模型名称到适配器的注册逻辑。
- `model_properties.json`：各模型最大上下文长度等属性。
- `impute_dataset.py`：读取 BM 缺失文件，调用填补算法并保存填补后的数据。
- `visualize_results.py`：旧版结果可视化入口。

### `Imputation/`

填补算法模块。

- `imputation_methods.py`：所有填补函数和 `IMPUTATION_METHODS` 注册表。
- `impute.py`：较早的窗口目录式填补接口，保留用于兼容旧流程。
- `README.md`：填补方法说明。

### `results/`

预测评估指标输出目录，按模型拆分。

- `results/<model>/clean/`：原始干净数据的预测结果。
- `results/<model>/impute/`：BM 缺失数据先填补再预测的结果。

结果文件通常包含 `MSE`、`MAE`、`sMAPE`、分位数预测相关指标等评估输出。

### `results_analysis/`

窗口特征分析输出目录。

- `results_analysis/<model>/prediction/`：预测窗口特征分析。
- `results_analysis/<model>/history/`：历史窗口特征分析。
- `results_analysis/<model>/overall_*_summary.json`：按模型汇总的分析摘要。
- `results_analysis/clean_prediction_windows/`：clean 预测窗口特征对照。

### `tools/`

工具脚本目录。

- `tools/Missing_Value_Injection/BM.py`：单数据集 BM 注入入口。
- `tools/Missing_Value_Injection/batch_bm_injection.py`：批量 BM 注入入口。
- `tools/Missing_Value_Injection/inject_range_utils.py`：根据数据集属性和预测周期计算注入区间。
- `tools/context_missing_ratio_report.py`：检查不同上下文区间内的缺失率。
- `tools/missing_ratio_checker.py`：缺失率检查工具。

### `文档资料/`

论文、实验文档和阶段性分析材料。目前包含文献资料、实验报告草稿和相关 PDF。

### 其他根目录文件

- `TSFIA.yml`：Conda 环境配置文件。
- `.gitignore`：Git 忽略规则。
- `.vscode/`、`.idea/`：本地 IDE 配置。
- `.kilo/`：本地工具相关目录，不属于核心实验代码。

## 常用命令

生成 BM 缺失数据：

```bash
python tools/Missing_Value_Injection/batch_bm_injection.py --missing_ratios 0.1,0.2,0.3 --block_length 50 --mode stratified
```

运行 clean 基线：

```bash
python Eval/run_eval.py clean --model chronos2 --dataset ETTh1 --term short
```

运行单个缺失文件的填补预测：

```bash
python Eval/run_eval.py single --model chronos2 --eval_data_path data/datasets/BM/BM_010/ETTh1_BM_length50_010_short.csv --imputation_method linear
```

批量运行缺失填补预测：

```bash
python Eval/run_batch_eval.py --model sundial --dataset ETTh1 --method BM --terms short,medium,long --missing_ratios 0.1,0.2,0.3 --imputation_methods mean,forward,backward,linear
```

运行窗口特征分析：

```bash
python Analysis/run_batch_analysis.py --model sundial chronos2 timesfm2p5 --terms short,medium,long --imputation_methods mean,forward,backward,linear
```

绘制按数据集比较的散点图：

```bash
python draw/visualized_results_by_dataset.py --model chronos2 --modes mean,forward,backward,linear
```

## 环境

推荐使用根目录 `TSFIA.yml` 创建环境：

```bash
conda env create -f TSFIA.yml
conda activate TSFIA
```

不同预测模型有额外依赖要求，例如 `chronos2` 需要 `chronos-forecasting`，`timesfm2p5` 需要包含 TimesFM 2.5 的 `timesfm`，`visiontspp` 需要 `visionts` 和 `huggingface_hub`，`saits` 填补需要 `pypots`。
