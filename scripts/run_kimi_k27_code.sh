#!/usr/bin/env bash
# 后台(脱离会话)启动 kimi-k2.7-code(经本地 gateway)全量 230 题生成,断点续跑。
# 用法:  bash run_doubao_lite.sh          # 全量 230 题
#         bash run_doubao_lite.sh 0        # 只跑 index 0(canary:先验证图片输入是否被接受)
# 隔离原理:doubao 走 gateway(与 minimax 主 .env 不同端点),用 DOTENV_PATH 指向 .env.gateway,
#           litellm.py / generate_explanation.py 的 load_dotenv 优先读它,不动主 .env。
#           见 [[eduillustrate-dotenv-override-collision]] / [[edubench-gateway-providers]]。
set -euo pipefail

cd /home/likefallwind/code/EduIllustrate
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

: "${API_GATEWAY:?API_GATEWAY 未设置(应在 ~/.bashrc export)}"
GW_BASE="${API_GATEWAY_BASE_URL:-http://127.0.0.1:8111/v1}"

# 基于主 .env 生成 gateway 版:保留其余配置(KOKORO 等),仅替换 CUSTOM_API_BASE/KEY。
grep -vE '^(CUSTOM_API_BASE|CUSTOM_API_KEY)=' .env > .env.gateway
{
  echo "CUSTOM_API_BASE=$GW_BASE"
  echo "CUSTOM_API_KEY=$API_GATEWAY"
} >> .env.gateway
export DOTENV_PATH="$(pwd)/.env.gateway"
# kimi-k2.7-code 上限 32768;否则 gateway 默认注入 max_tokens=65536 → HTTP 400 死循环重试。
export LITELLM_MAX_TOKENS=32768

mkdir -p output/kimi_k27_code
LOG="output/kimi_k27_code/run_$(date +%Y%m%d_%H%M%S).log"

INDEX_ARG=()
if [ "${1:-}" != "" ]; then
  INDEX_ARG=(--index "$1")
  LOG="output/kimi_k27_code/canary_${1}_$(date +%Y%m%d_%H%M%S).log"
fi

setsid python -u generate_explanation.py \
  --model "kimi-k2.7-code" \
  --problem_path data/benchmark/benchmark.json \
  --output_dir output/kimi_k27_code \
  --max_topic_concurrency 4 \
  --max_scene_concurrency 4 \
  --max_retries 3 \
  "${INDEX_ARG[@]}" \
  > "$LOG" 2>&1 < /dev/null &

PID=$!
disown || true
echo "已后台启动 kimi-k2.7-code (setsid),PID=$PID,endpoint=$GW_BASE"
echo "日志: $LOG"
echo "看进度: tail -f $LOG"
