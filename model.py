"""作业二模型占位模板；固定输出仅用于检查评测接口。"""
import numpy as np


class Model:
    def __init__(self):
        """由学生初始化模型参数与所需结构。"""
        pass

    def fit(self, X, y):
        """TODO：使用训练图像 X 和对应标签 y 实现模型训练。"""
        return None

    def predict(self, X):
        """TODO：替换占位结果，返回每张图像的预测类别。"""
        # 固定类别 0 仅用于跑通接口，不是模型或基线解法。
        return np.zeros(len(X), dtype=np.int64)
