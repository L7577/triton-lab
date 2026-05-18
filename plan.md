
# Triton 高并发推理服务实验计划书（阶段一～三）

## 1. 实验总目标

- 掌握 Triton 模型仓库、动态批处理、多实例并发的配置与优化方法。
- 量化在 GTX 1050 Ti (4GB) 上，不同并发策略对吞吐量、延迟、GPU 利用率的影响。
- 验证 HAMi 显存与算力分片在多服务共存时的隔离效果与性能损耗。
- 为后续多节点部署积累基准数据与配置经验。

## 2. 实验环境统一说明

| 项          | 说明                                                                   |
| ----------- | ---------------------------------------------------------------------- |
| GPU         | NVIDIA GTX 1050 Ti (4GB, 计算能力 6.1)                                 |
| 驱动        | ≥ 520.61.05（满足 Triton 22.12 要求）                                 |
| Triton 版本 | `nvcr.io/nvidia/tritonserver:22.12-py3`                              |
| 模型        | ResNet-18 ONNX（约 45MB），来自 ONNX Model Zoo                         |
| 压测工具    | Triton 自带的 `perf_analyzer`                                        |
| 监控工具    | `nvidia-smi`, Prometheus (Triton 端口 8002), HAMi 监控面板           |
| K8s 环境    | 已安装 HAMi-DRA，支持 `nvidia.com/gpumem` 和 `nvidia.com/gpucores` |
| 单机实验    | 使用 Docker 启动 Triton（实验一、二）                                  |

---

## 3. 实验一：单模型基准测试（理解基本性能）

### 3.1 目标

- 验证 Triton 基本功能，获得无任何优化时的“裸”推理延迟与吞吐。
- 建立性能基线，用于后续对比。

### 3.2 步骤

#### 3.2.1 准备模型仓库

创建目录结构：

```
/path/to/model_repository/
└── resnet18_onnx
    ├── 1
    │   └── model.onnx   # 下载的 ResNet-18 ONNX 文件
    └── config.pbtxt
```

`config.pbtxt` 内容（禁用批处理，单实例）：

```protobuf
name: "resnet18_onnx"
backend: "onnxruntime"
max_batch_size: 0
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

#### 3.2.2 启动 Triton Docker 容器

```bash
docker run --gpus all -it --rm \
  -v /path/to/model_repository:/models \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  nvcr.io/nvidia/tritonserver:22.12-py3 \
  tritonserver --model-repository=/models
```

观察日志，确认模型加载成功（显示 `READY`）。

#### 3.2.3 单并发延迟测试

```bash
perf_analyzer -m resnet18_onnx \
  --concurrency 1 \
  --shape data:3,224,224 \
  --measurement-interval 5000
```

记录 P50、P90、P99 延迟和吞吐量。

#### 3.2.4 固定低并发爬坡

分别测试 concurrency = 1, 2, 4，记录延迟和吞吐变化。此阶段无动态批处理，预期吞吐随并发线性增长有限。

#### 3.2.5 监控数据

- 通过 `curl localhost:8002/metrics` 或 Prometheus 抓取 `nv_inference_request_duration_us` 和 `nv_gpu_utilization`。
- 另开终端执行 `watch -n 1 nvidia-smi`，观察 GPU 利用率、显存占用。

### 3.3 实验一记录表

| 并发数 | 吞吐(infer/sec) | P50延迟(ms) | P99延迟(ms) | GPU利用率(%) | 显存占用(MB) |
| ------ | --------------- | ----------- | ----------- | ------------ | ------------ |
| 1      |                 |             |             |              |              |
| 2      |                 |             |             |              |              |
| 4      |                 |             |             |              |              |

---

## 4. 实验二：动态批处理与多实例并发优化

### 4.1 目标

- 开启动态批处理，观察吞吐量与延迟的变化。
- 增加多实例，进一步提升 GPU 占用率，测试极限并发能力。
- 找到在 4GB 显存下的最佳配置。

### 4.2 动态批处理实验

#### 4.2.1 修改配置

更新 `config.pbtxt` 后重新加载（Triton 支持动态加载，无需重启容器，只需将新配置放入仓库即可）：

```protobuf
name: "resnet18_onnx"
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

#### 4.2.2 并发爬坡测试

执行：

```bash
for c in 1 2 4 8 16; do
  perf_analyzer -m resnet18_onnx \
    --concurrency-range $c \
    --shape data:3,224,224 \
    --measurement-interval 10000
done
```

记录每个并发级别的吞吐与延迟。重点关注：

- 吞吐何时达到平台？
- P99 延迟在哪个并发点开始飙升？

#### 4.2.3 不同 max_batch_size 对比（可选）

将 `max_batch_size` 改为 8、32 重复测试，观察对显存和延迟的影响。4GB 显存下 `max_batch_size` 过大可能导致 OOM。

### 4.3 多实例实验

#### 4.3.1 修改配置

将实例数改为 2（注意显存）：

```protobuf
instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
```

保持动态批处理配置不变。

#### 4.3.2 再次执行爬坡测试

使用与 4.2.2 相同的命令。对比单实例的吞吐上限，多实例通常能提升 10%~30% 的吞吐，但延迟抖动可能增加。

#### 4.3.3 稳定性验证

使用恒定中等并发（例如 8）长时间运行 5 分钟，观察：

- 是否存在延迟毛刺？
- 显存是否稳定？有无 OOM？

### 4.4 实验二记录表

| 配置              | 并发数 | 吞吐 | P50/P99延迟 | GPU利用率 | 显存占用 |
| ----------------- | ------ | ---- | ----------- | --------- | -------- |
| 单实例+动态批处理 | 1~16   |      |             |           |          |
| 双实例+动态批处理 | 1~16   |      |             |           |          |

并绘制吞吐-并发曲线和延迟-并发曲线，标记最佳工作点。

---

## 5. 实验三：K8s 中 HAMi 分片多服务并发测试

### 5.1 目标

- 验证 HAMi 显存/算力分片下，多个 Triton 推理 Pod 共存时的性能隔离。
- 量化资源共享带来的性能损耗，特别是算力软隔离与显存带宽争抢。
- 为生产环境设置资源配额提供依据。

### 5.2 前置条件

- K8s 集群节点已安装 HAMi-DRA。
- 节点上只有一张 GTX 1050 Ti，确保不与其他工作负载冲突。
- 已制作好 Triton 镜像或使用官方镜像 `nvcr.io/nvidia/tritonserver:22.12-py3`，并在 Deployment 中挂载模型仓库（建议使用 hostPath 或 PVC）。

### 5.3 部署三个推理服务（Pod）

#### 5.3.1 模型仓库准备

在节点上存储三个独立的模型目录，分别为 `resnet18_onnx_a`, `resnet18_onnx_b`, `resnet18_onnx_c`，配置与实验二的“单实例+动态批处理”相同，但 `name` 需不同。

#### 5.3.2 创建 Deployment YAML 示例（Pod A）

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-a
spec:
  replicas: 1
  selector:
    matchLabels:
      app: triton-a
  template:
    metadata:
      labels:
        app: triton-a
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:22.12-py3
        args: ["tritonserver", "--model-repository=/models"]
        resources:
          limits:
            nvidia.com/gpumem: 2048   # 2GB 显存
            nvidia.com/gpucores: 50   # 50% 算力
        volumeMounts:
        - name: model-repo-a
          mountPath: /models
      volumes:
      - name: model-repo-a
        hostPath:
          path: /data/models/resnet18_onnx_a
```

类似地创建 Pod B（显存 1024, cores 30）和 Pod C（显存 1024, cores 20）。

#### 5.3.3 暴露服务

为每个 Deployment 创建 ClusterIP Service，或直接用 Pod IP + 端口 8001 测试。

### 5.4 性能对比测试方案

#### 5.4.1 单独运行基准

- 只启动 Pod A，用 `perf_analyzer` 从集群内客户端 Pod 压测：
  ```bash
  perf_analyzer -m resnet18_onnx_a --concurrency-range 1,2,4,8 -u <service-a:8001>
  ```

  记录单 Pod 独占 GPU 时的性能。

#### 5.4.2 三 Pod 共存空载

同时启动 Pod A、B、C，但不发送请求，通过 `nvidia-smi` 和 HAMi 面板确认显存被正确划分，三个模型均已加载。

#### 5.4.3 三 Pod 同时高负载

使用三个并行 `perf_analyzer` 进程，分别向三个服务发送固定并发（例如各 4 并发）。记录：

- 每个服务的吞吐量、P99 延迟。
- 通过 HAMi 监控查看每个 Pod 的算力使用率和显存使用率。
- 节点整体 GPU 利用率（从 `nvidia-smi`）。

#### 5.4.4 干扰验证

- 让 Pod A 跑满高并发，观察 Pod B 和 C 空载时的延迟变化（因算力争抢，空载服务也可能出现请求延迟增加）。
- 比较单独运行时的吞吐与共存高负载下的吞吐总和，计算性能损失百分比。

### 5.5 实验三记录表

| 场景            | Pod A (2GB/50%) | Pod B (1GB/30%) | Pod C (1GB/20%) | 总吞吐 | 节点GPU利用率 |
| --------------- | --------------- | --------------- | --------------- | ------ | ------------- |
| A单独满负载     | 吞吐/延迟       | -               | -               |        |               |
| 三Pod共存空载   | -               | -               | -               | -      |               |
| 三Pod共存高并发 | 吞吐/延迟       | 吞吐/延迟       | 吞吐/延迟       |        |               |

绘制柱状图对比单服务性能与共享后的性能，分析资源争抢对尾延迟的影响。

### 5.6 分析要点

- 算力限制是否是硬上限？高并发时能否达到分配值？
- 显存带宽竞争是否导致总吞吐不如裸卡？
- HAMi 提供的指标与实际负载关系是否一致？

---

## 6. 实验报告要点

所有实验完成后，需汇总：

- 动态批处理带来的吞吐提升倍数。
- 多实例与单实例的性能差异及显存开销。
- HAMi 分片在 1050 Ti 上的性能损耗百分比，以及适合的共享密度（例如建议单卡最多几个推理服务）。
- 对后续 10 节点 20 服务部署的配置建议：单卡同时运行的模型数量上限、推荐的资源配额分配策略。
