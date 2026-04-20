# Visualize 使用说明

本目录提供 `results_analysis` 结果的统一出图入口。

## 1. 功能概览

当前支持两类出图：

- `method`：同一模型、同一数据集、同一 term、同一缺失率(BM)下，把**所有填补方法**汇总到同一组图中
  - 历史窗口：6 张图（每个指标 1 张）
  - 预测窗口：6 张图（每个指标 1 张）
- `clean`：只比较干净窗口相关结果
  - 输出 6 张图（每个指标 1 张）
  - 仅比较：`clean history` vs `clean prediction ground truth`（不再使用模型 clean prediction 文件）
  - clean 模式不区分模型

统一入口脚本：`Visualize/cli.py`

批量入口脚本：`Visualize/batch_plot.py`

## 2. 依赖安装

至少需要：

- `matplotlib`
- `pandas`

可执行：

```bash
pip install matplotlib pandas
```

## 3. 如何执行

在仓库根目录执行。

### 3.1 多填补方法汇总对比（当前主要需求）

```bash
python Visualize/cli.py --plot-type method --model timesfm2p5 --dataset ETTh1 --term long --missing-ratio 010
```

可选布局参数：

- `--layout single`：每个指标单独一张图（默认也会输出）
- `--layout panel`：6 个指标合并到一张 2x3 总览图
- `--layout both`：同时输出 single + panel（默认）

说明：

- `--missing-ratio` 支持 `010` / `10` / `0.1`，会自动归一化为 `010`
- 不传 `--methods` 时，会自动扫描该 BM 下可用方法（如 `mean/linear/forward/backward`）

输出目录示例：

- `results_pic/method/timesfm2p5/ETTh1/long/BM_010/history/*.png`
- `results_pic/method/timesfm2p5/ETTh1/long/BM_010/prediction/*.png`

### 3.2 指定部分填补方法

```bash
python Visualize/cli.py --plot-type method --model timesfm2p5 --dataset ETTh1 --term long --missing-ratio 010 --methods mean,linear
```

### 3.3 纯干净窗口对比

```bash
python Visualize/cli.py --plot-type clean --dataset ETTh1 --term long
```

输出目录示例：

- `results_pic/clean/ETTh1/long/*.png`

clean 模式同样支持 `--layout single|panel|both`。

### 3.4 批量出图（method + clean）

默认会批量跑 `method,clean` 两种模式：

```bash
python Visualize/batch_plot.py
```

建议先加过滤条件做小范围验证：

```bash
python Visualize/batch_plot.py --models timesfm2p5 --datasets ETTh1 --terms long --missing-ratios 010 --plot-types method,clean
```

可用过滤参数：

- `--models`：模型过滤（仅 method 模式生效），逗号分隔
- `--datasets`：数据集过滤，逗号分隔
- `--terms`：term 过滤，逗号分隔
- `--missing-ratios`：仅 method 模式生效，缺失率过滤（如 `010,020,030`）
- `--plot-types`：`method` / `clean` / `method,clean`
- `--methods`：method 模式可选，固定方法列表（如 `mean,linear`）
- `--layout`：批量任务统一布局策略，`single|panel|both`

## 4. 参数说明

`python Visualize/cli.py -h` 可查看全部参数。

核心参数：

- `--plot-type`：`method` / `clean`
- `--model`：模型目录名（`results_analysis/{model}`，仅 `plot-type=method` 需要）
- `--dataset`：数据集名
- `--term`：`short` / `medium` / `long`
- `--missing-ratio`：仅 `plot-type=method` 需要
- `--methods`：可选，逗号分隔
- `--layout`：`single` / `panel` / `both`（默认）
- `--results-analysis-dir`：默认 `results_analysis`
- `--results-pic-dir`：默认 `results_pic`

## 5. 想添加“填补方法”时怎么做

### 5.1 常规新增（推荐）

只要 `results_analysis` 里有符合现有命名规范的 CSV，脚本会自动识别，无需改代码：

- 历史窗口：`{dataset}_BM_{ratio}_{term}_{method}_history.csv`
- 预测窗口：`{dataset}_BM_{ratio}_{term}_{method}_prediction.csv`

例如新增 `spline` 方法，只需确保文件名中 `{method}` 为 `spline`，并放在：

- `results_analysis/{model}/history/`
- `results_analysis/{model}/prediction/`

然后直接运行 `--plot-type method` 即可自动纳入对比。

### 5.2 临时只比较某些方法

使用 `--methods` 显式指定，不改代码：

```bash
--methods mean,linear,spline
```

## 6. 想增加新的“对比类型”时怎么做

当前代码已按“统一入口 + 路由 + 分类型模块”组织，扩展步骤：

1. 在 `Visualize/plots/` 下新增一个模块（如 `bm_compare.py` / `model_compare.py`）
2. 在模块里实现 `run_xxx_compare(...) -> List[Path]`
3. 在 `Visualize/router.py` 增加 `plot_type` 分支
4. 在 `Visualize/cli.py` 的 `--plot-type` 选项中加入新类型

这样可以持续扩展，而不把所有逻辑堆在一个文件。

## 7. 目录结构（当前）

```text
Visualize/
  cli.py
  batch_plot.py
  router.py
  config.py
  utils.py
  data_loader.py
  plotters.py
  plots/
    method_compare.py
    clean_compare.py
```
