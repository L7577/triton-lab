# Triton Lab — 推理服务性能基准测试

基于 [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) 的并发、动态批处理、多实例推理性能实验台。目标 GPU 为 GTX 1050 Ti (4GB)，使用自生成的 Identity ONNX 模型进行可重复的性能基准测试。

**本仓库只包含实验一和实验二**（Docker + 单卡 Triton）。实验三（K8s + HAMi-DRA 多 Pod 分片）位于独立仓库 [ai-inference-lab](https://github.com/L7577/ai-inference-lab)，运行环境与步骤完全不同，请勿在此尝试。

## 项目结构

```
triton-lab/
├── Makefile                       # 实验自动化（模型生成/容器管理/压测）
├── generate_onnx.py               # 生成 Identity ONNX 模型（148 bytes）
├── benchmark.py                   # Python HTTP 压测脚本
├── scripts/
│   └── run_experiments.sh         # Bash 版实验运行器
├── configs/
│   ├── exp1_baseline.pbtxt        # 实验一：单实例，无批处理
│   ├── exp2_batch.pbtxt           # 实验二a：单实例 + 动态批处理
│   └── exp2_multi.pbtxt           # 实验二b：双实例 + 动态批处理
├── model_repository/              # Triton 模型仓库
│   └── identity_onnx/
│       ├── 1/model.onnx           # ONNX 模型文件
│       └── config.pbtxt           # 当前活跃的 Triton 配置
├── plan.md                        # 原始实验计划（含实验三，仅作参考）
├── ADJUSTED-PLAN.md               # 调整后的实施方案（含实验三，仅作参考）
└── EXPERIMENT-REPORT.md           # 实验一、二的完整报告
```

> **注意**：`plan.md` 和 `ADJUSTED-PLAN.md` 描述了三阶段实验计划，其中实验三涉及 K8s/HAMi-DRA，不能在本仓库执行。此处保留作为设计上下文参考。实验内容以 [EXPERIMENT-REPORT.md](./EXPERIMENT-REPORT.md) 为准。

## 实验环境

| 项 | 说明 |
|---|---|
| GPU | NVIDIA GTX 1050 Ti (4GB) |
| 运行方式 | Docker 容器，不需要 Kubernetes |
| Triton 镜像 | `nvcr.io/nvidia/tritonserver:22.12-py3` |
| 模型 | 自生成 Identity ONNX（148 bytes，零下载） |
| 压测工具 | `benchmark.py`（Python 脚本，HTTP 调用） |
| 监控 | `nvidia-smi` + `curl localhost:8002/metrics` |

## 快速开始

```bash
# 前置条件
# - Docker + nvidia-container-toolkit
# - nvcr.io/nvidia/tritonserver:22.12-py3 镜像已拉取
# - pip install onnx numpy

# 一键运行实验一和实验二
make exp12

# 或分步执行
make model          # 生成 ONNX 模型 + 模型仓库
make exp1           # 实验一：基线（无批处理，单实例，并发 1/2/4）
make exp2-batch     # 实验二a：动态批处理（batch=16，单实例，并发 1/2/4/8/16）
make exp2-multi     # 实验二b：多实例（2 实例 + batch=16，并发 1/2/4/8/16）

# 调整每并发运行时长（默认 10 秒）
DURATION=30 make exp1

# 手动管理容器
make start          # 启动 Triton
make logs           # 查看日志
make stop           # 停止
make clean          # 清理产物
make help           # 查看所有命令
```

## 实验设计

| 实验 | 配置 | 目标 |
|------|------|------|
| 实验一 | `max_batch_size=0`, 1 GPU 实例 | 建立单实例无优化性能基线 |
| 实验二a | `max_batch_size=16`, dynamic batching, 1 实例 | 量化动态批处理对吞吐/延迟的影响 |
| 实验二b | `max_batch_size=16`, dynamic batching, 2 实例 | 量化多实例的边际收益与显存开销 |

## 为什么用 Identity ONNX 而非真实模型

- 模型仅 148 bytes，**零下载依赖**
- 计算量极小，瓶颈集中在 **Triton 调度行为**（排队、批处理、线程调度），而非 GPU 算力
- 可清晰观察 `max_batch_size`、`dynamic_batching`、`instance_group.count` 等配置项对吞吐和延迟的独立影响

## 关键发现

- 无批处理时吞吐随并发不升反降（43 → 37 req/s），单实例成为串行化瓶颈
- 动态批处理使吞吐随并发增长并稳定在平台（~40 req/s），但单并发延迟因排队略增
- 双实例吞吐提升约 6%，显存增加 25%——边际收益递减
- JSON 序列化是 hidden bottleneck，使用 gRPC 或 shared memory 可进一步提升

详见 [EXPERIMENT-REPORT.md](./EXPERIMENT-REPORT.md)。
