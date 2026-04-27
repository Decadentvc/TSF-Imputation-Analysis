# TSF-Imputation-Analysis

## 一、项目目标

来自当前研究主线：

- 研究维度：缺失模式 × 缺失程度 × 填补方法 × 时序基础模型
- 核心问题：在块状缺失场景下，不同填补策略如何改变 TSFM 的预测性能，以及这种影响是否通过改变历史窗口的趋势、季节、频谱结构发生。

## 二、当前实现与原设计的偏差

| 维度 | 根目录 README（已废弃）设想 | 代码实际支持 | 说明 |
| --- | --- | --- | --- |
| 缺失模式 | MCAR / BM / TM / TVMR | 仅 BM | `Eval/run_eval.py:25` 中 `ALLOWED_MISSING_METHODS={"BM"}`，`run_batch_eval.py --method` 也只允许 `BM` |
| 缺失率 | 5%~30% | 10% / 20% / 30% | `data/datasets/BM/` 下只有 `BM_010/020/030` |
| 块长度 | [1,3,5] | 固定 `length=50` | `tools/Missing_Value_Injection/BM.py` 默认 `block_length=50` |
| 填补方法 | todo | 全部为统计/插值方法 | `Imputation/imputation_methods.py` 当前包含 `zero`、`mean`、`forward`、`backward`、`linear`、`nearest`、`polynomial`、`spline`、`seasonal`，没有深度填补器（如 `SAITS`、`BRITS`、`CSDI`） |
| 基模 | todo | 已跑通 3 个，适配器多 4 个 | 已跑：`sundial`、`chronos2`、`timesfm2p5`；已适配但未跑：`kairos23m`、`kairos50m`、`timesfm2p0`、`visiontspp` |
| 不填补基线 | —— | 被禁用 | `run_eval.py:281` 的 missing 场景下 `imputation_method` 不能为 `none`；但 `SundialAdapter` 内部对 `NaN` 用 `LastValueImputation` 静默兜底（`model_adapters.py:127`） |

另外，仓库里存在两套填补接口，职责重叠：

1. `Imputation/impute.py`：基于窗口目录（`meta.json + window_XXX.csv`）的老版，没有被主流程使用。
2. `Eval/impute_dataset.py`：基于完整 CSV 的新版，被 `run_eval.py` 实际调用；输出路径为 `data/datasets/Imputed/BM/BM_{ratio}/{dataset}_BM_{ratio}_{term}_{method}.csv`。

文件名不包含 `block_length`，未来如果研究多块长，会互相覆盖。

## 三、已产出数据规模

- 原始数据：27 个数据集，位于 `data/datasets/ori/`，覆盖 ETT 系列、`electricity`、`traffic`、`weather`、`exchange_rate`、`national_illness`，以及多域采样频率对照（`Coastal_T_S` / `current_velocity` 各含 `5T/15T/20T/H`）。
- 注空数据：`data/datasets/BM/BM_{010,020,030}/`，`block_length=50`，`stratified` 模式。
- 填补数据：`data/datasets/Imputed/BM/BM_{010,020,030}/`，共 5 类填补方法。
- 评估结果：`results/{sundial,chronos2,timesfm2p5}/{clean,impute}/`。其中 `clean` 各 79 个，`impute` 中 `sundial` / `chronos2` 各 444 个，`timesfm2p5` 为 288 个。
- 中间预测：`data/Intermediate_Predictions/<model>/<eval_name>_prediction/[<impute_method>/]window_*.csv`
- 特征分析：`results_analysis/<model>/{prediction,history}/*.csv` 与 `overall_*_summary.json`；额外有 `results_analysis/clean_prediction_windows/` 保存 GT 窗口特征。
- 统计透视：`文档资料/_analysis_output.txt`（676 行，已完成 3 模型 × 26 数据集聚合）以及 `块状缺失对长序列预测影响-实验结果统计.csv`。

## 四、端到端使用流程

- 工作根目录：`TSF-Imputation-Analysis`
- 环境：参考 `TSFIA.yml`（`Python 3.10 + torch 2.11 + gluonts 0.16.2 + transformers 4.40.1 + timesfm 2.0 + statsmodels 0.14.6`）。
- 额外依赖：`chronos-forecasting>=2.1`（用于 `chronos2`）与包含 `timesfm_2p5` 的 `timesfm` 版本。

### 步骤 1：注入块缺失

若 `data/datasets/BM/` 已存在，可跳过。

```bash
# 单数据集
python tools/Missing_Value_Injection/BM.py --dataset ETTh1 --missing_ratio 0.1 --term long --no_auto_term

# 批量
python tools/Missing_Value_Injection/batch_bm_injection.py --missing_ratios 0.1,0.2,0.3 --block_length 50 --mode stratified
```

输出：

```text
data/datasets/BM/BM_{ratio}/{dataset}_BM_length50_{ratio}_{term}.csv
```

### 步骤 2：模型评估

`run_eval.py` 会在评估前检查 `data/datasets/Imputed/...`，若缺失则调用 `impute_dataset` 自动生成。

```bash
# 干净基线
python Eval/run_eval.py clean --model sundial --dataset ETTh1 --term short

# 单文件评估（指定填补方法）
python Eval/run_eval.py single --model chronos2 \
  --eval_data_path "data/datasets/BM/BM_010/ETTh1_BM_length50_010_short.csv" \
  --imputation_method linear

# 批量评估（推荐，支持断点续跑）
python Eval/run_batch_eval.py --model sundial --dataset ETTh1 --method BM \
  --terms short,medium,long \
  --imputation_methods linear,mean,forward,backward \
  --missing_ratios 0.1,0.2,0.3 \
  --device cuda:0 --batch_size 32
```

说明：

- 若省略 `--dataset`，则会遍历 `data/datasets/ori/` 下全部数据集。
- `--include_clean` 会同时跑 `clean`。
- `--clean_only` 仅跑 `clean`。
- `--force` 忽略已有结果。

输出：

- 指标 CSV：`results/<model>/{clean,impute}/*.csv`
- 逐窗口中间预测：`data/Intermediate_Predictions/<model>/<eval_name>_prediction/[<method>/]`

### 步骤 3：窗口特征与填补质量分析

```bash
# 预测窗口 + 历史窗口的 6 项 STL 特征（按模型批量）
python Analysis/run_batch_analysis.py --model sundial chronos2 timesfm2p5 \
  --terms short,medium,long \
  --imputation_methods linear,mean,forward,backward

# 干净 GT 窗口特征（作为对照）
python Analysis/run_clean_prediction_window_analysis.py --terms short,medium,long

# 填补保真度（整段比较，未区分 masked-only）
python Analysis/imputed_evaluation.py --output_dir results_analysis/imputed_evaluation
```

输出：

- `results_analysis/<model>/{prediction,history}/<combo>.csv` 与对应 `_summary.json`
- `results_analysis/<model>/overall_{prediction,history}_summary.json`
- `results_analysis/clean_prediction_windows/`
- `results_analysis/imputed_evaluation/*.json`

### 步骤 4：可视化

旧脚本位于 `Eval/visualize_results.py`：

```bash
python Eval/visualize_results.py --dataset ETTh1 --method BM --mode comparison \
  --imputation_methods none zero mean linear --format_mode new
```

注意：该脚本仍残留 `MCAR/TM/TVMR` 选项与 `none` 默认值，是“4 维矩阵”时期的产物；实际场景传 `--method BM` 即可。

## 五、既有实验揭示的核心结论

来自 `文档资料/_analysis_output.txt`：

1. 填补 `SMAPE` 与预测 `MSE` 的组内 `Spearman` 平均 `ρ=0.704`。在 156 个 `model × dataset × ratio` 分组中，89.7% 为正相关，说明填补越准，预测通常越好。
2. “填补冠军 = 预测冠军”的命中率为 66.7%，“预测垫底 = 填补垫底”的一致率为 74.4%。
3. 均值填补最差：在 47.4% 的组合中成为预测垫底；`Sundial × 均值 × 30%` 的平均退化率达到 80%。
4. 线性插值最稳：`chronos2`、`sundial`、`timesfm2p5` 的线性方法退化率均最低，其中 `sundial` 在 20% 缺失率时仅退化 2.6%。
5. 结构扰动相关性明显：`Sundial` 的 `|Δtrend_linearity|` 与预测退化之间 `ρ=0.757`，显著高于另外两个模型，支持“结构漂移作为中介变量”的假设。
6. 跨模型偏好并不一致：55.6% 的 `(dataset, ratio)` 组合中，3 个模型分别选出了不同的最佳填补方法。

## 六、主要风险与尚未补齐的工程点

1. `BM.py:230` 使用 Python 内置 `hash()` 生成 `seed` 偏移，跨进程不稳定，复现性存在风险。
2. `impute_dataset.py` 生成填补文件名时没有保留 `block_length`，未来多块长实验会互相覆盖。
3. `imputed_evaluation.py:236` 是整段比较而不是 `masked-only`，会稀释真实填补误差。
4. `metrics.py:186` 的 `spectral_entropy` 实现是 `Σ log(PSD)`，不是归一化谱熵，跨数据集不可比。
5. `SundialAdapter` 对 `NaN` 做了静默 `LastValueImputation` 兜底；如果要做“不填补 passthrough”基线，必须关闭或显式暴露该行为。
6. `eval_pipeline.py:156` 按列拆 `series` 后再平均，本质上是单变量预测结果的逐列均值；论文中应避免直接表述为 “multivariate forecasting”。
7. `results_analysis/<model>/overall_prediction_summary.json` 中 `imputed_combinations=0` 与实际存在的 imputed 分析文件不一致，暂时不能作为总汇总依据。

## 七、建议的下一步

1. 先把 `block_length` 写入填补文件名与结果路径，打开多块长对照实验。
2. 在 `imputation_methods.py` 中加入至少一个深度填补器（如 `SAITS` 或 `BRITS`），并允许 `none` 作为 passthrough 基线。
3. 将 `imputed_evaluation` 拆分为 `masked-only / unmasked` 两组指标。
4. 在 `Sundial / Chronos2` 层面暴露“是否禁用内部 NaN 兜底”的开关。
5. 固化 BM 随机种子，去除 `hash()` 依赖，并为每个注空配置记录至少 3 个 `seed`，以支持显著性检验。
