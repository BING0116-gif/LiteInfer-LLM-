"""Benchmark + Ablation 工具包（Task 12）。

三个脚本以"结果文件"为契约分层：
- ``liteinfer_benchmark`` 测量矩阵 -> results.csv + env.json（measure-only）
- ``benchmark_charts``    渲染图表 -> charts/*.png
- ``benchmark_report``    组装报告 -> report.md

把脚本声明为包（而不是散落的 .py 文件）是为了让 ``pytest`` 能以
``from benchmark.liteinfer_benchmark import ...`` 的方式单测每个组件；
运行方式不变：``set PYTHONPATH=d:/LiteInfer`` 后 ``python benchmark/xxx.py``。
"""