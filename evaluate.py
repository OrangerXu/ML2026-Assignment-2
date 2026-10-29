#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assignment 2 本地评测脚本（学生自评用）

用法（在仓库根目录，与 solution.py / model.py 同级）：

    python evaluate.py                          # 在 data/test_public.npz 上评测
    python evaluate.py --student-id 2026001 --name 张三
    python evaluate.py --rows 300               # 快速试跑前 300 张

流程：禁用库扫描（AST）-> 在隔离的子进程中加载你的 Solution ->
10 线程并发逐图推理 -> 计算 Accuracy / 混淆矩阵 / 耗时 ->
打印并写入 result.json。

本地自评使用子进程传递无标签图像，不构成操作系统安全沙箱。
教师另用容器核验器运行提交。

提交作业时提供 solution.py、model.py、analysis.py、report.md 和 result.json。
分析脚本和报告用于过程评价，不参与模型推理。
自评分数仅供参考，最终成绩以教师在保留数据上的复测为准。
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

EVALUATOR_VERSION = "2026.2.0"
ASSIGNMENT = "ML2026-Assignment-2"
DEFAULT_TEST = os.path.join("data", "test_public.npz")
CODE_FILES = ["solution.py", "model.py"]
N_CLASSES = 10

FORBIDDEN_LIBS = [
    "sklearn", "scikit-learn", "tensorflow", "torch", "pytorch",
    "keras", "xgboost", "lightgbm", "catboost", "statsmodels",
    "libsvm", "liblinear", "thundersvm", "cuml", "shotgun",
]


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_forbidden():
    hits = []
    for name in CODE_FILES:
        tree = ast.parse(Path(name).read_text(encoding="utf-8-sig"), filename=name)
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                      [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                if module.split(".")[0] in FORBIDDEN_LIBS:
                    hits.append("{}:{}: {}".format(name, node.lineno, module))
    return hits


# ---------------- 子进程：加载学生 Solution 并并发推理 ----------------

def _forward_all(X):
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        import solution
    except Exception as e:
        return None, 0.0, "导入 solution.py 失败: {}".format(e)
    try:
        sol = solution.Solution()
    except Exception as e:
        return None, 0.0, "初始化 Solution 失败: {}".format(e)

    samples = [X[i] for i in range(len(X))]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as ex:
        try:
            results = list(ex.map(sol.forward, samples))
        except Exception as e:
            return None, time.perf_counter() - t0, "forward 推理失败: {}".format(e)
    wall = time.perf_counter() - t0

    preds = [None] * len(results)
    for i, r in enumerate(results):
        try:
            preds[i] = int(r["prediction"])
        except Exception:
            return None, wall, \
                "第 {} 张图的返回值无法解析，应返回 {{'prediction': int}}，实际得到: {!r}".format(i, r)
    if any(p < 0 or p >= N_CLASSES for p in preds):
        return None, wall, "预测类别超出 0-9 范围"
    return preds, wall, None


def _worker_main(workdir):
    X = np.load(os.path.join(workdir, "features.npy"))
    preds, wall, error = _forward_all(X)
    out = {"predictions": preds,
           "info": {"prediction_time_s": round(wall, 3),
                    "training_and_setup_time_s": None, "error": error}}
    with open(os.path.join(workdir, "worker_out.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


# ---------------- 主进程：数据、指标与结果文件 ----------------

def _run_worker(X, timeout):
    workdir = tempfile.mkdtemp(prefix="ml2026_eval_")
    try:
        np.save(os.path.join(workdir, "features.npy"), X)
        cmd = [sys.executable, os.path.abspath(__file__), "--_worker", workdir]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="ignore",
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("推理超时（>{}s），请检查 forward 是否存在死循环或过慢".format(timeout))
        out_file = os.path.join(workdir, "worker_out.json")
        if not os.path.exists(out_file):
            tail = ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
            raise RuntimeError("推理子进程异常退出（code={}）：\n{}".format(
                proc.returncode, tail))
        with open(out_file, encoding="utf-8") as f:
            out = json.load(f)
        stdout_tail = (proc.stdout or "").strip()[-1500:]
        return out, stdout_tail
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def confusion(y_true, y_pred):
    m = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m


def main():
    ap = argparse.ArgumentParser(description="Assignment 2 本地评测")
    ap.add_argument("--test", default=DEFAULT_TEST, help="测试数据 npz 文件")
    ap.add_argument("--rows", type=int, default=0, help="只评测前 N 张（0=全部）")
    ap.add_argument("--out", default="result.json", help="结果输出文件")
    ap.add_argument("--student-id", default="", help="学号（可选）")
    ap.add_argument("--name", default="", help="姓名（可选）")
    ap.add_argument("--timeout", type=int, default=3600, help="推理阶段超时秒数")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        _worker_main(args._worker)
        return

    for name in CODE_FILES + [args.test]:
        if not Path(name).exists():
            sys.exit("缺少文件: {}（请在仓库根目录运行）".format(name))

    hits = scan_forbidden()
    if hits:
        sys.exit("[禁止] 检测到禁用库引用:\n  " + "\n  ".join(hits))

    d = np.load(args.test)
    X, y_true = d["X_test"], d["y_test"].astype(int)
    if args.rows:
        X, y_true = X[:args.rows], y_true[:args.rows]

    print("Evaluating {} images from {} ...".format(len(X), args.test))
    out, stdout_tail = _run_worker(X, args.timeout)

    error = out["info"].get("error")
    if error:
        sys.exit("[评测失败] {}".format(error))
    preds = out["predictions"]
    if preds is None or len(preds) != len(X):
        sys.exit("[评测失败] 预测数量与图像数量不一致")
    if stdout_tail:
        print("---- solution 输出 ----\n{}\n----------------------".format(stdout_tail))

    wall = out["info"]["prediction_time_s"]
    preds_arr = np.asarray(preds, dtype=int)
    accuracy = float((preds_arr == y_true).mean())
    cm = confusion(y_true, preds_arr)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    macro_recall = float(recall.mean())

    print("=" * 50)
    print("Accuracy:     {:.6f}".format(accuracy))
    print("Macro Recall: {:.6f}".format(macro_recall))
    print("Time:         {:.3f}s".format(wall))
    print("-" * 50)
    print("混淆矩阵 (行=真实, 列=预测):")
    header = "      " + " ".join("%4d" % j for j in range(N_CLASSES))
    print(header)
    for i in range(N_CLASSES):
        print("%4d  " % i + " ".join("%4d" % v for v in cm[i]))
    print("=" * 50)

    result = {
        "assignment": ASSIGNMENT,
        "evaluator_version": EVALUATOR_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "student": {"id": args.student_id, "name": args.name},
        "test_file": {"name": Path(args.test).name, "md5": md5_file(args.test),
                      "rows": int(len(X))},
        "metrics": {"accuracy": round(accuracy, 6), "macro_recall": round(macro_recall, 6)},
        "confusion_matrix": cm.tolist(),
        "prediction_time_s": wall,
        "code_md5": {name: md5_file(name) for name in CODE_FILES},
        "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                        for name in CODE_FILES},
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已写入 {}（提交作业时与代码一起上交）".format(args.out))


if __name__ == "__main__":
    main()
