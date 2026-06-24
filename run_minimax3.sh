#!/usr/bin/env bash
# 后台(脱离会话)启动 MiniMax-M3 全量 230 题生成,断点续跑。
# 用法:  bash run_minimax3.sh
set -euo pipefail

cd /home/likefallwind/code/EduIllustrate
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p output/minimax3
LOG="output/minimax3/run_$(date +%Y%m%d_%H%M%S).log"

setsid python -u generate_explanation.py \
  --model "MiniMax-M3" \
  --problem_path data/benchmark/benchmark.json \
  --output_dir output/minimax3 \
  --max_topic_concurrency 4 \
  --max_scene_concurrency 4 \
  --max_retries 3 \
  > "$LOG" 2>&1 < /dev/null &

PID=$!
disown || true
echo "已后台启动 (setsid),PID=$PID"
echo "日志: $LOG"
echo "看进度: tail -f $LOG"
