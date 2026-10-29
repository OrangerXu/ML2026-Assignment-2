"""作业二训练入口与单样本推理接口模板。"""
import numpy as np
from model import Model


class Solution:
    def __init__(self):
        """由学生完成 data/train.npz 的读取、预处理和模型训练。"""
        self.model = Model()

    def forward(self, sample: np.ndarray) -> dict:
        """输入为 (28, 28) 图像，返回 {'prediction': 0 至 9 的整数}。"""
        x = sample.reshape(1, -1)
        y_pred = self.model.predict(x)
        return {"prediction": int(y_pred[0])}
