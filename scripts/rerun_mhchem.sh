#!/usr/bin/env bash
# 重跑因缺 mhchem 而失败的化学题(mhchem 现已装好)。仅补环境,不改任何模型代码。
set -euo pipefail
cd /home/likefallwind/code/EduIllustrate
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
: "${API_GATEWAY:?API_GATEWAY 未设置}"
GW_BASE="${API_GATEWAY_BASE_URL:-http://127.0.0.1:8111/v1}"
grep -vE '^(CUSTOM_API_BASE|CUSTOM_API_KEY)=' .env > .env.gateway
{ echo "CUSTOM_API_BASE=$GW_BASE"; echo "CUSTOM_API_KEY=$API_GATEWAY"; } >> .env.gateway
export DOTENV_PATH="$(pwd)/.env.gateway"

declare -A DIRS=( [131]=problem_131_chemistry_g12 [150]=problem_150_chemistry_g9 )
LOG="output/doubao_lite/rerun_mhchem_$(date +%Y%m%d_%H%M%S).log"
{
  echo "=== rerun mhchem-blocked @ $(date) ==="
  for idx in 131 150; do
    echo ""; echo "############## index=$idx (${DIRS[$idx]}) ##############"
    PD="output/doubao_lite/${DIRS[$idx]}"
    rm -f "$PD"/scene*/render_failed.txt "$PD"/scene*/code/scene*_code_tokens.json
    echo "cleaned stale markers under $PD"
    python -u generate_explanation.py --model "doubao-seed-2.0-lite" \
      --problem_path data/benchmark/benchmark.json --output_dir output/doubao_lite \
      --max_scene_concurrency 4 --max_retries 3 --index "$idx" \
      || echo ">>> index=$idx 退出码非0"
  done
  echo ""; echo "=== 结束 @ $(date) ==="
} > "$LOG" 2>&1
echo "完成,日志: $LOG"
