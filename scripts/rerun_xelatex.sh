#!/usr/bin/env bash
# 重跑因缺 xelatex 而失败的 12 道(现已装好 xelatex+xeCJK+Noto CJK)。
# 断点续跑:已有 v3 的 ctex 代码会被直接重渲一次,通常无需重新生成代码。
# 隔离同 run_doubao_lite.sh:doubao 走 gateway(.env.gateway),不动主 .env。
set -euo pipefail

cd /home/likefallwind/code/EduIllustrate
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

: "${API_GATEWAY:?API_GATEWAY 未设置}"
GW_BASE="${API_GATEWAY_BASE_URL:-http://127.0.0.1:8111/v1}"

grep -vE '^(CUSTOM_API_BASE|CUSTOM_API_KEY)=' .env > .env.gateway
{
  echo "CUSTOM_API_BASE=$GW_BASE"
  echo "CUSTOM_API_KEY=$API_GATEWAY"
} >> .env.gateway
export DOTENV_PATH="$(pwd)/.env.gateway"

INDICES=(26 27 29 114 131 139 144 147 150 171 186 187)
# index -> problem dir 名(用于清理陈旧标记)
declare -A DIRS=(
  [26]=problem_26_biology_g12   [27]=problem_27_biology_g12   [29]=problem_29_biology_g12
  [114]=problem_114_geography_g9 [131]=problem_131_chemistry_g12 [139]=problem_139_biology_g12
  [144]=problem_144_biology_g12 [147]=problem_147_biology_g12 [150]=problem_150_chemistry_g9
  [171]=problem_171_biology_g9  [186]=problem_186_geography_g12 [187]=problem_187_geography_g12
)
LOG="output/doubao_lite/rerun_xelatex_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== rerun xelatex-blocked problems @ $(date) ==="
  echo "indices: ${INDICES[*]}"
  echo "endpoint: $GW_BASE"
  for idx in "${INDICES[@]}"; do
    echo ""
    echo "############## index=$idx (${DIRS[$idx]}) ##############"
    PD="output/doubao_lite/${DIRS[$idx]}"
    # 清理会导致增量竞态/旧失败态的标记;保留 scene1 现有 v3 代码以便直接重渲。
    rm -f "$PD"/scene*/render_failed.txt
    rm -f "$PD"/scene*/code/scene*_code_tokens.json
    echo "cleaned stale markers under $PD"
    python -u generate_explanation.py \
      --model "doubao-seed-2.0-lite" \
      --problem_path data/benchmark/benchmark.json \
      --output_dir output/doubao_lite \
      --max_scene_concurrency 4 \
      --max_retries 3 \
      --index "$idx" || echo ">>> index=$idx 退出码非0(见上方报错)"
  done
  echo ""
  echo "=== 全部重跑结束 @ $(date) ==="
} > "$LOG" 2>&1

echo "完成,日志: $LOG"
