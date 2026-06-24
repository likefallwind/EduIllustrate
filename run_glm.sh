#!/usr/bin/env bash
# 后台(脱离会话)启动 glm-5.2(经本地 gateway)生成,与 minimax3 并行、断点续跑。
# 用法:  bash run_glm.sh            # 全量 230 题
#         bash run_glm.sh 0         # 只跑 index 0(canary 验证)
# 隔离原理:DOTENV_PATH 指向 .env.gateway(主 .env 拷贝+换端点),
#           litellm.py / generate_explanation.py 的 load_dotenv 会优先读它,
#           不动主 .env,minimax3 完全不受影响。见 [[eduillustrate-dotenv-override-collision]]。
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

mkdir -p output/glm5
LOG="output/glm5/run_$(date +%Y%m%d_%H%M%S).log"

INDEX_ARG=()
if [ "${1:-}" != "" ]; then
  INDEX_ARG=(--index "$1")
  LOG="output/glm5/canary_${1}_$(date +%Y%m%d_%H%M%S).log"
fi

setsid python -u generate_explanation.py \
  --model "glm-5.2" \
  --problem_path data/benchmark/benchmark.json \
  --output_dir output/glm5 \
  --max_topic_concurrency 4 \
  --max_scene_concurrency 4 \
  --max_retries 3 \
  "${INDEX_ARG[@]}" \
  > "$LOG" 2>&1 < /dev/null &

PID=$!
disown || true
echo "已后台启动 glm-5.2 (setsid),PID=$PID,endpoint=$GW_BASE"
echo "日志: $LOG"
echo "$LOG"