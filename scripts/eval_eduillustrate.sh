#!/usr/bin/env bash
# 通用:对某个 EduIllustrate 全量生成结果用 LLM 判官评分,并写成 edubenchmark 报告。
# 非破坏:评测写新目录、报告写新目录,绝不删任何现有数据。可断点续(evaluate.py 复用 output_folder)。
#
# 用法:
#   bash eval_eduillustrate.sh <生成目录> <生成模型标签> [判官模型=MiniMax-M3] [题库=benchmark.json]
# 例:
#   bash eval_eduillustrate.sh output/minimax3  MiniMax-M3            # M3 生成、M3 判
#   bash eval_eduillustrate.sh output/doubao_lite doubao-seed-2.0-lite MiniMax-M3
#   bash eval_eduillustrate.sh output/kimi_k27_code kimi-k2.7-code   MiniMax-M3
#
# 判官端点路由:
#   - MiniMax-M3 / MiniMax-M2.7 → 直接用主 .env(已指向 minimax 官方端点),不改 .env。
#   - 其它(gateway 上的 glm/doubao/kimi/deepseek 等)→ 临时把主 .env 的 CUSTOM_API_BASE/KEY
#     改写到本地 gateway(备份 + 退出时还原),并设 LITELLM_MAX_TOKENS=32768(防 kimi 等被注入超限)。
# 可选环境变量:WORKERS(默认4) RETRY(默认2) API_GATEWAY API_GATEWAY_BASE_URL
set -u
cd /home/likefallwind/code/EduIllustrate
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GEN_DIR="${1:?用法: bash eval_eduillustrate.sh <生成目录> <生成模型标签> [判官] [题库]}"
GEN_LABEL="${2:?需要生成模型标签,例如 MiniMax-M3 / doubao-seed-2.0-lite / kimi-k2.7-code}"
JUDGE="${3:-MiniMax-M3}"
DATA="${4:-data/benchmark/benchmark.json}"
WORKERS="${WORKERS:-4}"
RETRY="${RETRY:-2}"

[ -d "$GEN_DIR" ] || { echo "生成目录不存在: $GEN_DIR" >&2; exit 1; }
EDUBENCH=/home/likefallwind/code/edubenchmark
slug(){ echo "$1" | sed 's#[^A-Za-z0-9._-]#-#g'; }
judge_slug="$(slug "$JUDGE")"; gen_base="$(basename "$GEN_DIR")"
EVAL_DIR="output/${gen_base}_eval_${judge_slug}"
REPORT_DIR="$EDUBENCH/reports/eval/eduillustrate/$(slug "$GEN_LABEL")__gen-full230_judge-${judge_slug}"
LOG="output/${gen_base}_eval_${judge_slug}_$(date +%Y%m%d_%H%M%S).log"

unset DOTENV_PATH
export STREAM_HEARTBEAT=60
export API_TIMING_LOG="output/_judge_${gen_base}_${judge_slug}_timing.log"

ENV_BAK=".env.evalbak.$$"
restore_env(){ [ -f "$ENV_BAK" ] && mv -f "$ENV_BAK" .env; }
case "$JUDGE" in
  MiniMax-M3|MiniMax-M2.7) ;;  # 主 .env 已是 minimax 官方端点,直接用
  *)
    : "${API_GATEWAY:?gateway 判官需 API_GATEWAY(应在 ~/.bashrc export)}"
    GW="${API_GATEWAY_BASE_URL:-http://127.0.0.1:8111/v1}"
    cp .env "$ENV_BAK"; trap restore_env EXIT
    grep -vE '^(CUSTOM_API_BASE|CUSTOM_API_KEY)=' "$ENV_BAK" > .env
    { echo "CUSTOM_API_BASE=$GW"; echo "CUSTOM_API_KEY=$API_GATEWAY"; } >> .env
    export LITELLM_MAX_TOKENS=32768
    ;;
esac

mkdir -p "$EVAL_DIR" "$REPORT_DIR"
echo "评测中…"
echo "  生成: $GEN_DIR (label=$GEN_LABEL) | 判官: $JUDGE | 题库: $DATA"
echo "  评测目录: $EVAL_DIR"
echo "  报告目录: $REPORT_DIR"
echo "  日志: $LOG"
{
  echo "=== eval $GEN_DIR (label=$GEN_LABEL) judged by $JUDGE @ $(date) ==="
  python -u evaluate.py --eval_type doc \
    --file_path "$GEN_DIR" --output_folder "$EVAL_DIR" \
    --model_doc "$JUDGE" --problem_data_path "$DATA" \
    --bulk_evaluate --combine --max_workers "$WORKERS" --retry_limit "$RETRY"
  echo "evaluate.py rc=$? @ $(date)"
  if compgen -G "$EVAL_DIR/evaluation_problem*.json" >/dev/null; then
    python "$EDUBENCH/scripts/eval/build_eduillustrate_report.py" \
      --source-repo "$(pwd)" --eval-dir "$EVAL_DIR" --gen-dir "$GEN_DIR" \
      --data-path "$(pwd)/$DATA" --model "$GEN_LABEL" --judge-model "$JUDGE" \
      --out "$REPORT_DIR"
    cp "$EVAL_DIR"/evaluation_problem*.json "$REPORT_DIR/" 2>/dev/null
    echo "report -> $REPORT_DIR"
  else
    echo "无 evaluation_problem*.json,跳过报告"
  fi
  echo "=== done @ $(date) ==="
} >> "$LOG" 2>&1
restore_env
echo "完成。结果见 $REPORT_DIR/summary.json"
