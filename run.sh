#!/bin/bash
# DeepSeek V3.2 启动脚本 - 8卡配置
# 模型路径: /mnt/shared/models/DeepSeek-V3.2

MODEL_PATH="/mnt/shared/models/DeepSeek-V3.2"
PORT=30000
HOST="0.0.0.0"

# 层数设置 (设置为空或0表示使用全部层)
# DeepSeek V3.2 原模型共61层
NUM_LAYERS="5"

# CUDA Graph 设置 (设置为1禁用，用于收集 decode 阶段数据)
# 注意: 禁用 CUDA graph 会降低性能，仅在需要调试/收集数据时使用
DISABLE_CUDA_GRAPH=""

# Warmup 设置 (默认跳过，设置为0启用 warmup)
# 注意: 跳过 warmup 会导致第一次推理延迟较高
SKIP_WARMUP="1"

# 构建额外参数
build_extra_args() {
    EXTRA_ARGS=""

    # 层数设置
    if [ -n "${NUM_LAYERS}" ] && [ "${NUM_LAYERS}" -gt 0 ] 2>/dev/null; then
        EXTRA_ARGS="${EXTRA_ARGS} --json-model-override-args {\"num_hidden_layers\":${NUM_LAYERS}}"
        echo ">>> 使用自定义层数: ${NUM_LAYERS} 层"
    else
        echo ">>> 使用全部层"
    fi

    # CUDA Graph 设置
    if [ "${DISABLE_CUDA_GRAPH}" = "1" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --disable-cuda-graph"
        echo ">>> CUDA Graph: 已禁用 (可收集 decode 数据，但性能降低)"
    else
        echo ">>> CUDA Graph: 已启用"
    fi

    # Warmup 设置
    if [ "${SKIP_WARMUP}" = "1" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --skip-server-warmup"
        echo ">>> Server Warmup: 已跳过"
    else
        echo ">>> Server Warmup: 已启用"
    fi
}


# ============================================
# 方式1: TP + DP Attention (推荐配置)
# 说明: DeepSeek V3.2 的 kernel 针对 dp_size=8 优化，推荐使用此配置
# ============================================
launch_tp_dp() {
    build_extra_args
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --trust-remote-code \
        --host ${HOST} \
        --port ${PORT} \
        ${EXTRA_ARGS}
}

# ============================================
# 方式2: Pure TP (无 DP)
# 说明: 纯 TP 模式，适合需要更简单配置的场景
# ============================================
launch_pure_tp() {
    build_extra_args
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp 8 \
        --trust-remote-code \
        --host ${HOST} \
        --port ${PORT} \
        ${EXTRA_ARGS}
}

# ============================================
# 方式3: EP + DP (Expert Parallelism)
# 说明: 使用 Expert Parallelism
# ============================================
launch_ep_dp() {
    build_extra_args
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp 8 \
        --ep 8 \
        --dp 8 \
        --enable-dp-attention \
        --trust-remote-code \
        --host ${HOST} \
        --port ${PORT} \
        ${EXTRA_ARGS}
}

# ============================================
# 方式4: TP + DP + MTP (Multi-Token Prediction)
# 说明: 启用 EAGLE 推测解码，可提升小 batch 解码速度
# ============================================
launch_with_mtp() {
    build_extra_args
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --trust-remote-code \
        --host ${HOST} \
        --port ${PORT} \
        --speculative-algorithm EAGLE \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        ${EXTRA_ARGS}
}

# ============================================
# 方式5: 带 Function Calling 和 Reasoning Parser
# ============================================
launch_with_tools() {
    build_extra_args
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --trust-remote-code \
        --host ${HOST} \
        --port ${PORT} \
        --tool-call-parser deepseekv32 \
        --reasoning-parser deepseek-v3 \
        ${EXTRA_ARGS}
}

# ============================================
# 主入口 - 选择启动方式
# ============================================
case "${1:-tp_dp}" in
    "tp_dp")
        echo "启动方式: TP + DP Attention (推荐)"
        launch_tp_dp
        ;;
    "pure_tp")
        echo "启动方式: Pure TP"
        launch_pure_tp
        ;;
    "ep_dp")
        echo "启动方式: EP + DP"
        launch_ep_dp
        ;;
    "mtp")
        echo "启动方式: TP + DP + MTP"
        launch_with_mtp
        ;;
    "tools")
        echo "启动方式: 带 Function Calling"
        launch_with_tools
        ;;
    *)
        echo "用法: $0 [tp_dp|pure_tp|ep_dp|mtp|tools]"
        echo ""
        echo "启动模式:"
        echo "  tp_dp   - TP + DP Attention (推荐，默认)"
        echo "  pure_tp - 纯 TP 模式"
        echo "  ep_dp   - Expert Parallelism + DP"
        echo "  mtp     - 启用 Multi-Token Prediction"
        echo "  tools   - 带 Function Calling 和 Reasoning Parser"
        echo ""
        echo "配置项 (修改脚本顶部变量):"
        echo ""
        echo "  NUM_LAYERS        - 控制加载的层数"
        echo "                      例如: NUM_LAYERS=10 只加载前10层"
        echo "                      留空或设为0使用全部层 (DeepSeek V3.2 共61层)"
        echo ""
        echo "  DISABLE_CUDA_GRAPH - 禁用 CUDA Graph"
        echo "                      设为1禁用，用于收集 decode 阶段数据"
        echo "                      注意: 禁用会降低推理性能"
        echo ""
        echo "  SKIP_WARMUP        - 跳过 Server Warmup (默认: 1)"
        echo "                      设为1跳过，设为0启用"
        echo "                      注意: 跳过会导致首次推理延迟较高"
        exit 1
        ;;
esac
