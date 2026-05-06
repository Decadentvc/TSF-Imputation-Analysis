# Imputation Methods

本目录提供时间序列缺失值填补方法。主流程实际调用
`Eval/impute_dataset.py`，底层算法统一注册在 `Imputation/imputation_methods.py`。

## 接口约定

每个填补函数遵循同一接口：

```python
method(df: pd.DataFrame, data_cols: list, ...) -> pd.DataFrame
```

- `df`：包含缺失值的完整数据表，时间列通常已被设为 index。
- `data_cols`：需要填补的数值列。
- 返回值：填补后的 `DataFrame`，保留原 index 和列名。
- 随机方法支持 `random_seed` 参数，默认 `42`。

`Eval/impute_dataset.py` 会自动读取 CSV、识别时间列、调用算法并保存到：

```text
data/datasets/Imputed/{method}/{method}_{ratio}/{dataset}_{method}_{ratio}_{term}_{imputation}.csv
```

## 当前方法

### 基础方法

- `none`：不填补，保留 NaN。
- `zero`：用 0 填补。
- `mean`：用列均值填补。
- `forward`：前向填补。
- `backward`：后向填补。
- `linear`：线性插值。
- `nearest`：最近邻插值。
- `polynomial`：二阶多项式插值。
- `spline`：三阶样条插值。
- `seasonal`：基于季节分解的填补，失败时回退到线性插值。

### 新增强单序列方法

- `kalman_struct`

  使用局部线性趋势状态空间模型和 Kalman smoother。状态包含 level 和
  slope，适合趋势明显、块状缺失较长的单变量序列。该实现只依赖
  NumPy/Pandas。

- `kalman_arima`

  先从初始插值序列估计稳定 AR(p) 系数，再用 AR 状态空间 Kalman
  smoother 填补缺失值。默认 `max_lag=3`，用于近似 ARIMA 动态。该实现只
  依赖 NumPy/Pandas。

- `stl_kalman`

  先估计趋势和季节项，再对残差使用 `kalman_struct` 填补，最后重构序列。
  如果安装了 `statsmodels`，会优先使用 `STL`；否则使用季节 profile 和滚动
  趋势作为轻量回退。适合检验“结构保持型填补”对下游 TSFM 的影响。

- `gp_rbf`

  使用一维时间索引上的 RBF Gaussian Process 填补缺失点。长序列会抽取不
  超过 `max_train_points=512` 个观测点以控制矩阵求解开销；抽样由
  `random_seed` 控制，默认 `42`。该实现只依赖 NumPy/Pandas。

- `saits`

  使用 PyPOTS 中的 SAITS 自注意力填补模型。为了保持单序列设定，当前实现
  逐列训练和填补，不使用跨列信息。需要额外安装：

  ```bash
  python -m pip install pypots
  ```

  默认参数偏轻量：`n_steps=96`、`epochs=10`、`batch_size=32`、`device="cpu"`。
  CPU 默认值用于保证固定 `random_seed` 时尽量可复现。大规模正式实验前建议
  根据数据集长度和 GPU 情况调大 `epochs`，如显式使用 CUDA，则还需要注意
  PyTorch/CuBLAS 的确定性设置。

## 使用示例

单文件填补：

```bash
python Eval/impute_dataset.py \
  --eval_data_path data/datasets/BM/BM_010/ETTh1_BM_length50_010_short.csv \
  --imputation_method kalman_struct \
  --base_output_dir data/datasets/Imputed
```

带随机种子：

```bash
python Eval/impute_dataset.py \
  --eval_data_path data/datasets/BM/BM_010/ETTh1_BM_length50_010_short.csv \
  --imputation_method gp_rbf \
  --random_seed 42
```

批量评估时直接传入新方法：

```bash
python Eval/run_batch_eval.py \
  --model chronos2 \
  --dataset ETTh1 \
  --method BM \
  --terms short \
  --missing_ratios 0.10 \
  --imputation_methods kalman_struct,kalman_arima,stl_kalman,gp_rbf \
  --random_seed 42
```

## 依赖说明

- 必需：`numpy`、`pandas`。
- 推荐：`statsmodels`，用于 `stl_kalman` 的 STL 分解。
- SAITS：需要 `pypots`，通常也会依赖 `torch`。

当前实现对缺失依赖采用显式处理：`saits` 在缺少 PyPOTS 时会抛出安装提示；
`stl_kalman` 在缺少 statsmodels 时会使用内置轻量季节分解回退。若当前系统
用户目录不可写，`saits` 会把 PyPOTS 的生态配置目录临时指向当前工作目录下
的 `.pypots_home`。
