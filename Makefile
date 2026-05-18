.PHONY: model clean start stop logs \
        exp1 exp2-batch exp2-multi exp2 exp12 \
        help

MODEL_REPO  := model_repository
IMAGE      := nvcr.io/nvidia/tritonserver:22.12-py3
CONTAINER  := triton-lab
PORT_HTTP  := 8000
BENCH      := python3 benchmark.py
BENCH_URL  := http://localhost:$(PORT_HTTP)/v2/models/identity_onnx/infer
DURATION   ?= 10

# ============================================
# Model
# ============================================
model:
	python3 generate_onnx.py identity.onnx
	@mkdir -p $(MODEL_REPO)/identity_onnx/1
	cp identity.onnx $(MODEL_REPO)/identity_onnx/1/model.onnx
	cp configs/exp1_baseline.pbtxt $(MODEL_REPO)/identity_onnx/config.pbtxt
	@echo "Model repository ready: $(MODEL_REPO)/identity_onnx"

# ============================================
# Triton lifecycle
# ============================================
start: stop
	@docker run -d --rm --gpus all \
		--name $(CONTAINER) \
		-v $(CURDIR)/$(MODEL_REPO):/models \
		-p $(PORT_HTTP):8000 -p 8001:8001 -p 8002:8002 \
		$(IMAGE) \
		tritonserver --model-repository=/models
	@echo -n "Waiting for model READY..."
	@for i in $$(seq 1 30); do \
		if curl -s -o /dev/null -w "%{http_code}" http://localhost:$(PORT_HTTP)/v2/models/identity_onnx/ready 2>/dev/null | grep -q 200; then \
			echo " READY ($${i}s)"; exit 0; \
		fi; \
		sleep 1; \
		echo -n "."; \
	done
	@echo " FAILED"; docker logs --tail 20 $(CONTAINER); exit 1

stop:
	@docker stop $(CONTAINER) 2>/dev/null || true

logs:
	@docker logs $(CONTAINER) 2>&1 | grep -iE "successfully|failed|READY|error|instance" || docker logs --tail 30 $(CONTAINER)

# ============================================
# Experiments
# ============================================
_exp1_config:
	cp configs/exp1_baseline.pbtxt $(MODEL_REPO)/identity_onnx/config.pbtxt

_exp2_batch_config:
	cp configs/exp2_batch.pbtxt $(MODEL_REPO)/identity_onnx/config.pbtxt

_exp2_multi_config:
	cp configs/exp2_multi.pbtxt $(MODEL_REPO)/identity_onnx/config.pbtxt

exp1: model _exp1_config start
	@echo; echo "=== Experiment 1: Baseline (no batching, 1 instance) ==="; echo
	$(BENCH) --url $(BENCH_URL) --model identity_onnx --sweep 1,2,4 --duration $(DURATION)
	@echo; echo "--- GPU State ---"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
	@echo; echo "--- Triton Metrics ---"; curl -s http://localhost:8002/metrics 2>/dev/null | grep -E "nv_inference_request_success|nv_inference_request_failure" | head -3
	@$(MAKE) --no-print-directory stop

exp2-batch: model _exp2_batch_config start
	@echo; echo "=== Experiment 2a: Dynamic Batching (batch=16, 1 instance) ==="; echo
	$(BENCH) --url $(BENCH_URL) --model identity_onnx --sweep 1,2,4,8,16 --duration $(DURATION)
	@echo; echo "--- GPU State ---"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
	@$(MAKE) --no-print-directory stop

exp2-multi: model _exp2_multi_config start
	@echo; echo "=== Experiment 2b: Multi-Instance (2) + Dynamic Batching ==="; echo
	$(BENCH) --url $(BENCH_URL) --model identity_onnx --sweep 1,2,4,8,16 --duration $(DURATION)
	@echo; echo "--- GPU State ---"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
	@$(MAKE) --no-print-directory stop

exp2: exp2-batch exp2-multi

exp12: exp1 exp2
	@echo; echo "=== Experiments 1 & 2 complete ==="; echo "Results in EXPERIMENT-REPORT.md"

# ============================================
# Cleanup
# ============================================
clean: stop
	@rm -f identity.onnx
	@rm -rf $(MODEL_REPO)

# ============================================
# Help
# ============================================
help:
	@echo "Triton Lab — Inference Experiments"
	@echo ""
	@echo "  make model         Generate ONNX model + model repo"
	@echo "  make start         Start Triton container"
	@echo "  make stop          Stop Triton container"
	@echo "  make logs          Show Triton logs"
	@echo ""
	@echo "  make exp1          Experiment 1: Baseline"
	@echo "  make exp2-batch    Experiment 2a: Dynamic Batching"
	@echo "  make exp2-multi    Experiment 2b: Multi-Instance"
	@echo "  make exp2          Experiment 2: both parts"
	@echo "  make exp12         Experiments 1 & 2"
	@echo ""
	@echo "  make clean         Stop Triton + remove artifacts"
	@echo ""
	@echo "  DURATION=15 make exp1   Override benchmark duration (default 10s)"
