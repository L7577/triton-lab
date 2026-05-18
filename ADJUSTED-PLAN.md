# Triton 推理实验 — 调整后方案

## 设计原则

- **最小化制品**: 自生成 1KB 以内的 ONNX 模型，零下载
- **最简工具链**: Python 脚本直接打 Triton HTTP API，不装 perf_analyzer
- **复用已有基础设施**: 实验三直接使用 ai-inference-lab 的 k8s + HAMi-DRA 模式

## 环境

| 项 | 说明 |
|---|---|
| GPU | GTX 1050 Ti (4GB) |
| Triton | `nvcr.io/nvidia/tritonserver:22.12-py3` |
| 模型 | 自生成 Identity ONNX (~1KB) |
| 压测 | Python 脚本 → Triton HTTP `/v2/models/.../infer` |
| 监控 | `nvidia-smi` + `curl :8002/metrics` |

---

## 实验一：单模型基准

**目标**: 验证 Triton 基本功能，获得无优化基线

**配置**: `max_batch_size: 0`, `count: 1`, `kind: KIND_GPU`

**步骤**:
1. 生成 identity.onnx，创建模型仓库
2. 启动 Triton Docker 容器
3. 并发 1, 2, 4 各压测 30s
4. 记录吞吐、P50/P99 延迟、GPU 利用率、显存

---

## 实验二：动态批处理与多实例

**目标**: 量化 dynamic batching 和 multi-instance 的吞吐提升

**配置**:
- 单实例 + dynamic_batching (max_batch_size: 16, delay: 100us)
- 双实例 + dynamic_batching

**步骤**:
1. 更新 config.pbtxt 开启动态批处理
2. 并发 1, 2, 4, 8, 16 爬坡
3. 改为双实例，重复爬坡
4. 绘制吞吐-延迟曲线，找饱和点

---

## 实验三：K8s HAMi-DRA 多服务并发

**目标**: 量化 GPU 分片下的性能隔离与共享损耗

**使用**: ai-inference-lab 已有 k8s + HAMi-DRA 基础设施

**Pod 配置**:
| Pod | cores | memory |
|---|---|---|
| model-high | 40 | 1600Mi |
| model-mid | 35 | 1200Mi |
| model-low | 25 | 800Mi |

**步骤**:
1. 各 Pod 单独满负载 → 获得 solo 吞吐基线
2. 三 Pod 同时高并发 → 对比吞吐总和
3. 干扰测试: A 满负载，观察 B、C 空载延迟
4. 计算共享开销: (solo_sum - concurrent_sum) / solo_sum
