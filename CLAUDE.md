# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EduIllustrate generates illustrated, step-by-step explanation documents for STEM problems. Given a problem (text + optional image), it plans an outline, generates Manim code per diagram scene, renders each scene to a PNG, and assembles a Markdown `solution.md`. A separate evaluation suite scores generated documents across 8 dimensions using an LLM judge.

## Environment setup

Always run with the repo root on `PYTHONPATH` and the venv active:

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
```

API credentials live in `.env` (copy from `.env.template`). The system depends on heavy native libs (FFmpeg, LaTeX, Cairo/Pango) for Manim rendering — see README for the apt/brew install lists.

LaTeX rendering of Chinese/chemistry scenes needs packages a basic TeX install lacks, and a missing one surfaces as a per-scene render failure (not an install error): **xelatex** (`texlive-xetex`) — the model emits CJK scenes via a `ctex`/`xeCJK` template with `tex_compiler="xelatex"`; **mhchem** (`texlive-science`) — chemistry `\ce{...}`. CJK rendering also needs a Noto/Source-Han CJK font installed.

## Common commands

Generate an explanation (single problem index):

```bash
python generate_explanation.py --model "gpt-5" \
  --problem_path data/benchmark/benchmark.json \
  --output_dir output/my_run --index 0 --max_retries 3 --translate_to_chinese
```

Evaluate generated docs:

```bash
python evaluate.py --eval_type doc \
  --file_path output/my_run --output_folder output/eval \
  --model_doc "gpt-5" --bulk_evaluate --combine --max_workers 4 \
  --problem_data_path data/benchmark/benchmark.json
```

Smoke testing: use `data/benchmark/smoke5.json` (5-problem subset; gitignored). There is **no `tests/` directory** despite the README's mention of `pytest tests/` — that command does not work. Verification is done by running the pipeline end-to-end against the smoke set (see `output/smoke_*.log` for prior runs).

Regenerate prompt modules after editing any `*.txt` prompt: the `prompts_raw/__init__.py` files are **generated**, not hand-edited. Run the matching `parse_prompt.py` to rebuild them:

```bash
python task_generator/parse_prompt.py   # rebuilds task_generator/prompts_raw/__init__.py
python eval_suite/parse_prompt.py        # rebuilds eval_suite/prompts_raw/__init__.py
```

Each `.txt` becomes a module-level string variable named `_<filename>` (e.g. `prompt_code_generation.txt` → `_prompt_code_generation`), imported elsewhere by that name.

## Architecture

### Generation pipeline (`generate_explanation.py` orchestrates)

Three core stages in `src/core/`, all async:

1. **`ExplanationPlanner`** (`explanation_planner.py`) — `generate_scene_outline()` decomposes a problem into interleaved `<TEXT_k>` blocks and `<SCENE_k>` diagram descriptions (parsed via `parse_scene_outline_tokens`). Then generates per-scene implementation plans, optionally concurrently.
2. **`CodeGenerator`** (`code_generator.py`) — `generate_manim_code()` turns a scene plan into Manim code. Two strategies: **incremental (default)** plans Scene 1 in detail and uses its code as a style reference for later scenes; **all-parallel** plans every scene independently. `fix_code_errors()` and `visual_self_reflection()` drive the retry/repair loop.
3. **`ExplanationRenderer`** (`explanation_renderer.py`) — runs `manim -pql -s` (low quality, save last frame), exports each scene's final frame as a PNG, and `combine_explanations()` assembles text + images into the doc.

`parse_explanation.py` extracts images/frames from rendered output. Optional Chinese translation runs as a final pass preserving LaTeX.

Key behaviors: `--max_scene_concurrency` parallelizes scenes within one problem; `--max_topic_concurrency` parallelizes problems. `--use_visual_fix_code` feeds rendered images back to a VLM for repair. `--use_rag` enables retrieval from a Chroma vector store (`src/rag/`). `--only_plan`/`--disable_code` short-circuit code generation.

### LLM access (`mllm_tools/`)

`LiteLLMWrapper` (`litellm.py`) is the single async LLM entry point (`__call__` takes a list of `{"type": "text"|"image"|..., "content": ...}` messages). It formats multimodal payloads differently per model family (gemini / gpt / generic OpenAI-compatible). Setting `CUSTOM_API_BASE` + `CUSTOM_API_KEY` in `.env` routes **all** calls through an OpenAI-compatible custom endpoint — this is how Kimi, MiniMax, Doubao, DeepSeek, Qwen etc. are served. `gemini.py` / `vertex_ai.py` are alternate provider wrappers.

### Allowed-models gate

`--model` and `--helper_model` choices are restricted to the list in `src/utils/allowed_models.json`. **A new model must be added there** or argparse will reject it. (Recent commits do exactly this for doubao/deepseek.)

### Evaluation suite (`evaluate.py` + `eval_suite/`)

`--eval_type doc` scores a generated doc on 8 dimensions (4 text-only, 2 text-diagram synergy, 2 diagram-only) via an LLM judge, using the rubric in `eval_suite/doc_evaluation_rubric.json`; overall score is the geometric mean. Other eval types (`text`, `explanation`, `image`, `all`) exist for finer-grained scoring. `doc_utils.py`, `text_utils.py`, `image_utils.py`, `explanation_utils.py` hold the per-dimension logic. Judge timing logs land in `output/_judge_*_timing.log`.

### MCP server (`mcp_server.py`)

A FastMCP server exposing the generator as tools (`generate_diagram_and_text`, `list_available_models`, `png_to_base64`). See `MCP_SERVER_README.md`.

### Other

- `src/config/config.py` — `Config` holds default paths (output dir, RAG paths, embedding model, Kokoro TTS settings).
- `annotation_app/` — standalone Flask/static app for human annotation of results.
- `task_generator/prompts_raw/` — the full Manim prompt library (cheatsheets, scene planning, error-fix, RAG query generation, etc.).
- Output layout per problem: `output/<run>/<problem_name>/{doc/, scene<k>/{code,media,prompt.json}, timing.json}`.

## Operational gotchas

- **Single endpoint per process.** Both `litellm.py` and `evaluate.py` call `load_dotenv(override=True)`, so all calls in one run share the `.env`'s `CUSTOM_API_BASE`/`CUSTOM_API_KEY`. To run a model on a *different* endpoint without touching the main `.env`, point `DOTENV_PATH` at an alternate env file (`litellm.py` honors it; `generate_explanation.py` uses this to route non-MiniMax models through a local gateway). You cannot drive two endpoints from the same checkout concurrently — use a separate env file or a git worktree.
- **Re-running a previously failed problem.** Generation resumes per scene (a scene with `succ_rendered.txt` is skipped). But the incremental strategy makes Scene 2+ wait on Scene 1, detecting "Scene 1 done" via `scene1/code/scene1_code_tokens.json`; if a stale copy of that file exists with no `succ_rendered.txt`, the topic aborts in ~0.01s with `Scene 1 processing finished but render failed` before Scene 1 can re-render. To genuinely retry a failed problem, first delete its `scene*/render_failed.txt` and `scene*/code/scene*_code_tokens.json` (keep the existing `*_v*.py` to re-render without regenerating code).
- **Evaluation/reporting lives in a separate repo.** Generation writes only to `output/`; it never writes to the benchmark repo. The sibling `edubenchmark` repo consumes EduIllustrate output via `scripts/eval/run_eduillustrate_judges.sh` (runs `evaluate.py` on a substrate, swapping the LLM-as-judge) and `scripts/eval/build_eduillustrate_report.py` (turns `evaluation_problem*.json` into `summary.json`/`scored.jsonl`/`report.html`).

## Conventions

- Generation, code-gen, rendering, and the LLM wrapper are all `async`; the CLI drives them via `asyncio`.
- Token usage is tracked through the wrapper (`get_token_usage()` / `reset_token_usage()`); see `TOKEN_TRACKING_README.md`.
- `output/`, `.env`, `.venv`, `models/`, `*.log`, and `data/benchmark/smoke*.json` are gitignored.
