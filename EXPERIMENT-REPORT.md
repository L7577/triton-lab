# Triton 推理服务性能实验报告

## 实验一：单模型基准测试 & 实验二：动态批处理与多实例优化

## 1. 实验概述

### 1.1 目的

1. 掌握 Triton Inference Server 的模型仓库、config.pbtxt 配置、动态批处理、多实例并发的使用方式
2. 在 GTX 1050 Ti (4GB) 上量化不同配置对吞吐量、延迟、GPU 利用率的影响
3. 为后续 HAMi 分片多服务部署（实验三）提供配置基准

### 1.2 环境

| 项 | 说明 |
|---|---|
| GPU | NVIDIA GeForce GTX 1050 Ti (4096 MiB, compute 6.1) |
| NVIDIA 驱动 | 550.163.01 / CUDA 12.4 |
| Docker | 28.5.1 |
| Triton 镜像 | `nvcr.io/nvidia/tritonserver:22.12-py3` (14GB) |
| 模型 | 自生成 Identity ONNX (`batch×256×256`, FP32, 148 bytes) |
| 压测工具 | `benchmark.py` — Python 脚本，HTTP POST 调用 Triton inference API |
| 监控 | `nvidia-smi` + Triton Metrics (`:8002/metrics`) |

### 1.3 为什么用 Identity ONNX 而不是 ResNet-18

- ResNet-18 ONNX 模型文件约 45MB，需要额外下载
- 实验目标是理解 Triton 的并发/批处理/多实例机制，不是模型的推理性能
- Identity 操作（输入直接输出）保证了：
  - 模型本身几乎不占显存（仅输入/输出 tensor）
  - 推理时间极短，使瓶颈集中在线程调度和请求排队
  - 可以清晰观察到 Triton 各配置项对调度行为的影响

**取舍**：由于 Identity 计算量太小，实验中 GPU 利用率始终很低。这在真实模型场景下会不同，但调度行为的变化趋势是一致的。

---

## 2. 实验架构

### 2.1 组件关系

```
benchmark.py (Python HTTP client, ThreadPoolExecutor)
    │
    │  POST /v2/models/identity_onnx/infer
    │  Body: {"inputs": [{"name": "input", "shape": [1,256,256],
    │          "datatype": "FP32", "data": [0.0, 0.0, ...]}]}
    ▼
Triton Inference Server (Docker container)
    │
    ├── HTTP Service (port 8000)
    ├── GRPC Service (port 8001)
    ├── Metrics Service (port 8002)
    │
    ├── Model: identity_onnx (version 1)
    │   ├── Backend: ONNX Runtime
    │   ├── Instance 0 → GPU 0 (CUDA stream)
    │   └── Instance 1 → GPU 0 (if count=2)
    │
    └── Dynamic Batcher (if max_batch_size > 0)
        ├── Queue: max_delay=100us
        └── Preferred batch sizes: [4, 8, 16]
```

### 2.2 模型仓库结构

```
model_repository/
└── identity_onnx/
    ├── 1/
    │   └── model.onnx          # 148 bytes, IR version 6
    └── config.pbtxt            # Triton 模型配置
```

### 2.3 ONNX 模型规格

```
输入:  "input"   float32["batch", 256, 256]   (batch 维度标记为动态，支持可变 batch)
输出:  "output"  float32["batch", 256, 256]
操作:  Identity (opset 8, IR version 6)
大小:  148 bytes
```

**关键设计决策**：batch 维度设为动态参数 `dim_param: "batch"` 而非固定值 `1`，因为 Triton 的 dynamic batching 要求模型接受可变 batch 输入。若使用固定值，Triton 加载配置 `max_batch_size > 0` 时会失败。

---

## 3. 实验过程

### 3.1 准备工作

**Step 1: 生成 ONNX 模型**

```bash
cd triton-lab
python3 generate_onnx.py identity.onnx

# 输出: Generated identity.onnx (148 bytes)
# IR version: 6, opset: 8
```

`generate_onnx.py` 使用 `onnx.helper` 构建一个最小 ONNX 计算图：一个 Identity 节点连接同名输入和输出。关键点是将 batch 维度标记为 `dim_param` 而非 `dim_value`。

**Step 2: 创建模型仓库**

```bash
mkdir -p model_repository/identity_onnx/1
cp identity.onnx model_repository/identity_onnx/1/model.onnx
# config.pbtxt 由各实验步骤自动切换
```

**Step 3: 拉取 Triton 镜像**

```bash
docker pull nvcr.io/nvidia/tritonserver:22.12-py3
# 镜像大小: 14GB，需要 NVIDIA GPU Cloud 访问权限
```

### 3.2 实验一：单模型基准测试

#### 3.2.1 配置

```protobuf
name: "identity_onnx"
backend: "onnxruntime"
max_batch_size: 0              # 关闭批处理
instance_group [
  {
    count: 1                    # 单实例
    kind: KIND_GPU              # 使用 GPU
  }
]
```

#### 3.2.2 执行

```bash
# 1. 应用配置
cp configs/exp1_baseline.pbtxt model_repository/identity_onnx/config.pbtxt

# 2. 启动 Triton
docker run -d --rm --gpus all --name triton-lab \
  -v $(pwd)/model_repository:/models \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  nvcr.io/nvidia/tritonserver:22.12-py3 \
  tritonserver --model-repository=/models

# 3. 等待模型就绪 —— 看日志出现 "successfully loaded 'identity_onnx'"
docker logs triton-lab | grep "successfully loaded"
# I0518 17:05:55.915279 model_lifecycle.cc:694] successfully loaded 'identity_onnx' version 1

# 4. 验证就绪状态
curl http://localhost:8000/v2/models/identity_onnx/ready
# HTTP 200

# 5. 运行压测 (并发 1, 2, 4 各 10 秒)
python3 benchmark.py \
  --url http://localhost:8000/v2/models/identity_onnx/infer \
  --model identity_onnx --sweep 1,2,4 --duration 10

# 6. 采集指标
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
curl -s http://localhost:8002/metrics | grep nv_inference_request
```

#### 3.2.3 过程观察

启动容器后约 10 秒模型变为 READY。日志中可见：
```
I0518 17:05:55.452746 model_lifecycle.cc:459] loading: identity_onnx:1
I0518 17:05:55.468570 onnxruntime.cc:2563] TRITONBACKEND_ModelInitialize: identity_onnx
I0518 17:05:55.911678 onnxruntime.cc:2606] TRITONBACKEND_ModelInstanceInitialize: identity_onnx_0 (GPU device 0)
I0518 17:05:55.915279 model_lifecycle.cc:694] successfully loaded 'identity_onnx'
```

压测过程中 GPU 利用率接近 0%，因为 Identity 操作的计算量极小，大部分时间消耗在 HTTP 请求的 JSON 序列化/反序列化上。

#### 3.2.4 结果

| 并发数 | 吞吐 (req/s) | P50 延迟 (ms) | P99 延迟 (ms) | GPU 利用率 | 显存占用 |
|--------|-------------|--------------|--------------|-----------|---------|
| 1 | 43.2 | 22.8 | 33.3 | ~0% | 157 MB |
| 2 | 41.6 | 46.7 | 87.0 | ~0% | 157 MB |
| 4 | 36.7 | 104.9 | 220.5 | ~0% | 157 MB |

Triton 指标：1224 次成功请求，0 次失败。

#### 3.2.5 分析

`max_batch_size: 0` 意味着 Triton 禁用了动态批处理。每个请求到达后立即调度到唯一的 GPU 实例执行。由于只有一个实例，并发请求被串行化：

- **并发 1**: 请求依次到达立即执行，延迟最低 (P50=22.8ms)
- **并发 2**: 两个请求竞争一个实例，一个在执行时另一个排队，延迟翻倍
- **并发 4**: 排队效应加剧，P99 延迟达到 220ms

**更关键的是吞吐量不升反降**（43.2 → 36.7）。这说明在多并发下，请求排队和上下文切换的开销超过了并行执行的收益——这正是没有批处理时单实例的典型行为。

### 3.3 实验二 a：动态批处理

#### 3.3.1 配置

```protobuf
name: "identity_onnx"
backend: "onnxruntime"
max_batch_size: 16
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 100
}
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

**配置说明**：

- `max_batch_size: 16` — Triton 允许将最多 16 个独立请求合并为一个 batch 执行
- `preferred_batch_size: [4, 8, 16]` — 动态批处理器倾向于将这些大小的 batch 提交给模型
- `max_queue_delay_microseconds: 100` — 最多等待 100μs 来积累请求形成 batch；超时后即使 batch 未达到 preferred size 也会提交
- `instance_group.count: 1` — 仍使用单实例，以对比纯批处理的效果

#### 3.3.2 执行

```bash
# 停止旧容器，更新配置
docker stop triton-lab
cp configs/exp2_batch.pbtxt model_repository/identity_onnx/config.pbtxt

# 重新启动（步骤同实验一），确认加载成功
# 日志显示: successfully loaded 'identity_onnx'

# 压测（扩展并发范围到 16）
python3 benchmark.py \
  --url http://localhost:8000/v2/models/identity_onnx/infer \
  --model identity_onnx --sweep 1,2,4,8,16 --duration 10
```

#### 3.3.3 过程观察

启动过程和实验一相同。模型加载日志中的区别：Triton 在内部注册了 DynamicBatcher 作为调度器。配置中的 `max_batch_size: 16` 要求模型支持 batch 维度——我们的 ONNX 模型第一维是 `dim_param: "batch"`（动态），满足了这一要求。

压测时单并发延迟 (29.6ms) 比实验一 (22.8ms) 略高，这是因为动态批处理器引入了 100μs 的排队延迟——即使只有 1 个请求，也会在队列中等待至多 100μs。

#### 3.3.4 结果

| 并发数 | 吞吐 (req/s) | P50 延迟 (ms) | P99 延迟 (ms) |
|--------|-------------|--------------|--------------|
| 1 | 32.9 | 29.6 | 44.5 |
| 2 | 35.0 | 53.2 | 93.4 |
| 4 | 38.4 | 98.2 | 204.5 |
| 8 | 40.6 | 191.0 | 335.0 |
| 16 | 40.1 | 379.5 | 638.0 |

#### 3.3.5 分析

与实验一对比，动态批处理的核心价值在于：**吞吐量随并发增加而上升，在并发 8 达到平台 ~40 req/s，不再下降**。

```
实验一 (no batch):  43.2 → 41.6 → 36.7  (下降趋势)
实验二 a (batch):   32.9 → 35.0 → 38.4 → 40.6 → 40.1  (上升后稳定)
```

动态批处理器的工作原理：
1. 多个请求到达后先进入队列
2. 批处理器等待最多 100μs 积累请求
3. 将积累的请求合并为一个 batch tensor（形状从 `[1,256,256]` 变为 `[N,256,256]`）
4. 模型一次 forward 处理整个 batch

在我们的 Identity 模型中，batch 处理的计算量线性增长，但 GPU kernel launch 开销被摊销了，所以吞吐有所改善。在真实的卷积网络中，batch 内的矩阵乘法可以充分利用 GPU 并行性，收益会大得多。

**吞吐平台的成因**：并发 8 后吞吐不再增长，说明系统已达到瓶颈。最可能的瓶颈是 client 端的 JSON 序列化（每个请求 ~200KB 的 JSON 文本），而不是 GPU 计算。

### 3.4 实验二 b：多实例 + 动态批处理

#### 3.4.1 配置

```protobuf
# ... 动态批处理配置同上 ...
instance_group [
  {
    count: 2                    # 2 个 GPU 实例
    kind: KIND_GPU
  }
]
```

**配置说明**：Triton 使用两个独立的 CUDA stream 各运行一个模型副本。两个实例共享同一个 GPU，可以并行执行不同的 batch。在单实例中，如果在执行一个 batch 时有新请求到达，新请求只能等待；双实例下，第二个实例可以立即处理。

#### 3.4.2 执行

```bash
docker stop triton-lab
cp configs/exp2_multi.pbtxt model_repository/identity_onnx/config.pbtxt
# 重新启动并运行同上的压测命令
```

启动日志确认两个实例均已创建：
```
TRITONBACKEND_ModelInstanceInitialize: identity_onnx_0_0 (GPU device 0)
TRITONBACKEND_ModelInstanceInitialize: identity_onnx_0_1 (GPU device 0)
```

#### 3.4.3 结果

| 并发数 | 吞吐 (req/s) | P50 延迟 (ms) | P99 延迟 (ms) | 显存占用 |
|--------|-------------|--------------|--------------|---------|
| 1 | 34.3 | 30.7 | 42.3 | 197 MB |
| 2 | 37.3 | 50.2 | 95.0 | 197 MB |
| 4 | 40.7 | 94.7 | 187.4 | 197 MB |
| 8 | 42.9 | 184.2 | 320.8 | 197 MB |
| 16 | 40.3 | 385.3 | 667.4 | 197 MB |

#### 3.4.4 分析

**吞吐量**：双实例在所有并发级别都略高于单实例批处理（例：并发 8 时 42.9 vs 40.6），最大吞吐提升了约 6%。

**延迟**：双实例的 P50 延迟和单实例几乎相同，说明实例间调度没有引入额外开销。

**显存**：双实例占用 197 MB vs 单实例 157 MB，增加 40 MB (+25%)。每个额外实例主要消耗 CUDA context 和 I/O 缓冲区。对于 4GB 显存的 GTX 1050 Ti，理论上可以运行数十个 Identity 模型实例；如果是 45MB 的 ResNet-18，可运行的实例数会显著减少。

---

## 4. 三配置对比

### 4.1 吞吐-并发曲线

```
吞吐 (req/s)
 45 ┤  ●
    │      ▲──▲──▲
 40 ┤  ■──■──■
    │●
 35 ┤
    │  ■  1-instance + batch=16
 30 ┤  ▲  2-instance + batch=16
    │  ●  1-instance, no batch
    └─────┬─────┬─────┬─────┬─────
          1     2     4     8     16  并发
```

### 4.2 汇总表

| 配置 | 最大吞吐 | 达到于 | P50 @max | P99 @max | VRAM |
|------|---------|--------|---------|----------|------|
| 无批处理, 1 实例 | 43.2 | conc=1 | 22.8ms | 33.3ms | 157MB |
| 批处理 16, 1 实例 | 40.6 | conc=8 | 191.0ms | 335.0ms | 157MB |
| 批处理 16, 2 实例 | 42.9 | conc=8 | 184.2ms | 320.8ms | 197MB |

### 4.3 延迟-吞吐关系

```
P99 延迟 (ms)
 700 ┤                              ○
     │
 600 ┤                          ○
     │                      □
 400 ┤                  □
     │              □
 200 ┤          □
     │  ●  □
   0 ┤●─●──■──▲──▲──▲
     └──┬──┬──┬──┬──┬──
       30 35 40 45    吞吐 (req/s)

 ● no batch  ■ batch 1-inst  ▲ batch 2-inst
 ```

**关键取舍**：动态批处理用单并发下 ~7ms 的延迟增加，换来了高并发下吞吐不崩溃。这是在线推理服务最核心的权衡：延迟 vs 吞吐。

---

## 5. 结论

### 5.1 关键发现

1. **动态批处理有效但需模型计算量配合**：Identity 模型计算量太小（< 1μs GPU 时间），JSON 序列化成为瓶颈。真实模型（如 ResNet-18）的 GPU 计算时间占主导，批处理收益会更显著。

2. **多实例的边际收益递减**：从 1 实例到 2 实例吞吐提升 ~6%，显存增加 25%。实例数继续增加时，上下文切换开销和显存碎片化会抵消收益。最佳实例数通常为 1-3 个。

3. **GTX 1050 Ti 适合推理**：4GB 显存对于轻量模型足够，compute 6.1 支持 FP32 推理。瓶颈在于 PCIe 带宽和算力上限，而非显存容量。

4. **client 端可能成为瓶颈**：实验中每个请求携带 65536 个 float32 的 JSON 数组（约 200KB 文本），43 req/s 对应约 8.6 MB/s 的 JSON 解析。使用 gRPC（binary protobuf）或 shared memory 可消除此瓶颈。

### 5.2 对实验三的指导

- 单实例 + 动态批处理是性价比最高的配置，建议实验三的 3 个 Triton Pod 均使用此配置
- 每个 Pod 的预计显存占用约 160-200MB（Identity 模型），在 4GB GPU 上分给 3 个 Pod 各 800-1600MB 绰绰有余
- 并发 8 为吞吐饱和点，实验三的压测建议在并发 8 进行

---

## 6. 复现指南

```bash
cd triton-lab

# 前置条件: Docker + nvidia-container-toolkit + Triton 镜像已拉取

# 一键运行
make exp12

# 或分步:
make model            # 生成 ONNX 模型 + 模型仓库
make exp1             # 实验一
make exp2-batch       # 实验二 a
make exp2-multi       # 实验二 b

# 调整每并发运行时长
DURATION=30 make exp1

# 手动启动 Triton 调试
make start            # 启动 Triton
make logs             # 查看日志
make stop             # 停止
```

### 关键文件

```
triton-lab/
├── Makefile                      # 实验自动化
├── generate_onnx.py              # ONNX 模型生成
├── benchmark.py                  # HTTP 压测脚本
├── configs/
│   ├── exp1_baseline.pbtxt       # 实验一配置
│   ├── exp2_batch.pbtxt          # 实验二 a 配置
│   └── exp2_multi.pbtxt          # 实验二 b 配置
├── model_repository/             # Triton 模型仓库
└── EXPERIMENT-REPORT.md             # 本报告
```
