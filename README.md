# Assignment 2：手写体分类任务

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

本作业要求对 28×28 灰度手写数字图像进行分类，独立实现核支持向量机（Kernel SVM），完成核函数对比、混淆矩阵分析与类别不均衡处理实验。

## 1. 任务概述与作业要求

预测目标为数字类别，任务类型为监督学习多分类。输入为 28×28 灰度图像，像素值范围为 0 至 1；输出为 0 至 9 的整数标签。模型训练数据为 `data/train.npz`，公开验证数据为 `data/test_public.npz`。

| 数据文件 | 样本数量 | 标签 | 用途 |
|---|---:|---|---|
| `data/train.npz` | 6800 | 包含 `y_train` | 模型训练与训练集内验证 |
| `data/test_public.npz` | 1000 | 包含 `y_test` | 本地自评、方案比较与误差分析 |

需完成以下内容：

1. **模型实现。** 在 `model.py` 和 `solution.py` 中独立实现 Kernel SVM 核心算法，包括训练逻辑、核函数、多分类策略和预测过程。
2. **核函数对比。** 比较多项式核与高斯核（RBF）的效果。实验代码保存在 `analysis.py`，报告中说明预处理、参数设置与对照结果。
3. **误差分析与改进。** 根据混淆矩阵分析易混淆类别及稀有类别的召回表现，开展类别不均衡处理或特征预处理实验。未取得改善的实验也应如实记录。
4. **结果复现。** 固定训练与分析中的随机种子，提交完整自评结果及相应源码，保证教师能够重新训练和评测。

模型核心算法可以使用 NumPy、pandas 和 Python 标准库。禁止直接导入或调用以下机器学习库及其现成模型：

`sklearn` / `scikit-learn`、`tensorflow`、`torch` / `pytorch`、`keras`、`xgboost`、`lightgbm`、`catboost`、`statsmodels`、`libsvm`、`liblinear`、`thundersvm`、`cuml`、`shotgun`。

`analysis.py` 可以使用 `matplotlib` 等分析与可视化库，不参与模型推理。核函数对比和混淆矩阵分析应基于提交的模型，不得另用现成模型替代作业核心算法。

## 2. 数据格式说明

数据采用 NumPy `.npz` 格式存储。`N` 表示样本数量，各数组的含义如下：

| 数组 | 形状 | 含义与取值说明 |
|---|---|---|
| `X_train` | `(6800, 28, 28)` | 训练图像，像素值范围为 0 至 1 |
| `y_train` | `(6800,)` | 训练标签，取值为 0 至 9 的整数 |
| `X_test` | `(1000, 28, 28)` | 公开验证图像，像素值范围为 0 至 1 |
| `y_test` | `(1000,)` | 公开验证标签，取值为 0 至 9 的整数 |

## 3. 评测指标与评分方式

### 3.1 评测指标

| 指标 | 含义 | 作用 |
|---|---|---|
| Accuracy | 分类准确率 | 效果评分的主要指标，越高越好 |
| Macro Recall | 各类召回率的平均 | 观察模型在不同类别上的整体召回表现 |
| 混淆矩阵 | 真实类别与预测类别的交叉统计 | 分析易混淆类别和分类错误 |
| 耗时 | 训练与准备、预测耗时 | 检查实现的运行成本 |

公开验证集指标用于本地检查与方案比较。

### 3.2 评分方式

- Accuracy 越高，模型表现越好。

参考基线成绩：Accuracy 65.4%（`data/test_public.npz`）。

## 4. 项目结构

```text
ML2026-Assignment-2/
├── data/
│   ├── train.npz          # 训练数据
│   └── test_public.npz    # 公开验证数据
├── model.py               # 模型核心算法，需修改
├── solution.py            # 训练入口与推理接口，需修改
├── evaluate.py            # 本地自评脚本
├── requirements.txt       # 本地实验环境依赖
└── README.md              # 作业说明
```

提交时还需提供自行编写的 `analysis.py`、`report.md` 及评测生成的 `result.json`。

## 5. 模型实现与接口规范

### 5.1 模型训练

建议在 `Solution.__init__` 中读取 `data/train.npz`，完成特征预处理和模型训练，将参数保存在内存中。`model.py` 用于组织 SVM 核心算法，`solution.py` 用于连接训练流程与评测接口。

模型的数据来源仅限 `data/train.npz`。公开验证标签不得并入训练数据，不得读取其他测试文件、联网获取额外数据、加载外部训练产物或在源码中硬编码预训练权重和答案映射。

### 5.2 单样本推理接口

评测器调用 `Solution.forward(sample)`。输入为形状 `(28, 28)` 的 NumPy 数组，不包含标签；输出必须包含 `prediction`，其值为 0 至 9 的整数类别。

```python
import numpy as np

class Solution:
    def forward(self, sample: np.ndarray) -> dict:
        """接收单张 28x28 图像，返回预测类别。"""
        x = sample.reshape(1, -1)
        y_pred = self.model.predict(x)
        return {"prediction": int(y_pred[0])}
```

评测器使用 10 个线程并发调用同一个 `Solution` 实例。`forward` 不应重新训练模型或修改共享权重，训练与推理的特征处理规则应保持一致。

教师运行模型时仅加载 `solution.py` 和 `model.py`，运行环境提供 NumPy、pandas 和标准库，不提供 SciPy 等额外第三方库。运行所需源码应放在这两个文件中，不得依赖额外模块或个人计算机上的绝对路径。每次公开或私有评测的预算为 1800 秒，包含训练与预测；内存上限为 4 GiB。

## 6. 环境配置

建议使用 Python 3.11。

```bash
conda create -n ML2026 python=3.11
conda activate ML2026
pip install -r requirements.txt
```

`requirements.txt` 包含本地分析与可视化所需依赖，库使用范围以第 1 节为准。

## 7. 本地评测

在项目根目录执行：

```bash
# 完整评测公开验证集
python evaluate.py

# 快速检查前 300 张
python evaluate.py --rows 300

# 在结果文件中记录学号和姓名
python evaluate.py --student-id 20260001 --name 张三
```

评测结束后，脚本输出 Accuracy、Macro Recall 和混淆矩阵，并生成 `result.json`，记录评测器版本、数据摘要、样本数量、源码摘要、环境信息及耗时。`--rows` 用于调试；提交前应对最终源码执行完整评测。

修改源码后必须重新生成结果文件。

## 8. 提交要求

提交以下五个文件，保持文件名不变并采用 UTF-8 编码：

| 文件 | 提交内容 |
|---|---|
| `solution.py` | 模型训练入口与单样本预测接口 |
| `model.py` | SVM 核心算法实现 |
| `analysis.py` | 核对比实验与混淆矩阵分析代码 |
| `report.md` | 模型说明、核对比结论、混淆矩阵分析、类别不均衡处理及 AI 工具使用声明 |
| `result.json` | 最终源码的完整自评结果 |

报告应说明模型与训练方法、预处理步骤、核函数对比设置与结果、混淆矩阵分析、类别不均衡处理，以及参考资料和 AI 工具的使用情况。实验结果应与提交源码对应。

## 9. 学术诚信与参考资料

允许查阅算法资料，使用情况应在报告中注明；使用 AI 工具时，应说明辅助范围。

复制历年解答、窃取或硬编码测试标签、探测教师评测数据、伪造实验结果以及其他违反数据与库使用规则的行为，将按课程规则核查处理。
