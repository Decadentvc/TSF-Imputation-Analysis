# TSF-Imputation-Analysis

### 准备数据
从 [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)、[[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) 或 [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library) 下载预处理数据。

实验从以下 4 个维度/层次展开：

> 缺失模式 → 缺失程度 → 填充方法 → 时序预测基模

---

## 1. 缺失模式（Missing Pattern）

采用以下几种缺失模式：

1. **全随机缺失**
2. **块/连续缺失（块数[1,3,5]）**
3. **阈值缺失**
4. **缺失率/缺失密度随时间变化的缺失（[递增，递减],[线性，非线性]）**

---

## 2. 缺失程度（Missing Ratio）

总缺失率[5%，10%，15%，20%，25%，30%]

---

## 3. 填充方法（Imputation Methods）

todo

---

## 4. 时序预测基模（Base Forecasting Models）

todo