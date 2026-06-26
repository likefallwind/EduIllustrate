# EduIllustrate

English | [简体中文](README_CN.md)

## 📖 Overview

**EduIllustrate** is an AI-powered educational diagram generation system that automatically creates detailed illustrated explanations for problems in mathematics, physics, chemistry, biology, and other subjects.

The system takes problem descriptions and images as input, and through a multi-stage planning, code generation, and rendering pipeline, produces:
- 📝 Structured Markdown explanation documents
- 🎨 High-quality diagrams rendered with Manim
- 🌐 Support for bilingual output (English and Chinese)

## ✨ Key Features

- 🤖 **Multi-Model Support**: Compatible with OpenAI GPT, Anthropic Claude, Google Gemini, Moonshot Kimi, and other mainstream LLMs
- 🎬 **Professional Diagrams**: Generates educational-grade mathematical/physical/chemical animation diagrams using Manim
- 📊 **Multi-Dimensional Evaluation**: Built-in 8-dimensional document quality assessment system
- 🔄 **Auto-Retry**: Intelligent error detection and code fixing mechanism
- ⚡ **Concurrent Processing**: Supports both scene-level and problem-level concurrency for improved efficiency
- 🌍 **Smart Translation**: One-click translation to Chinese while preserving all LaTeX formulas and formatting

## 🏗️ Architecture

```
EduIllustrate
├── generate_explanation.py    # Main generation script
├── evaluate.py                 # Evaluation script
├── src/
│   ├── core/                  # Core modules
│   │   ├── explanation_planner.py      # Explanation planner
│   │   ├── code_generator.py           # Code generator
│   │   ├── explanation_renderer.py     # Renderer
│   │   └── parse_explanation.py        # Result parser
│   ├── config/                # Configuration management
│   ├── rag/                   # RAG retrieval augmentation
│   └── utils/                 # Utility functions
├── eval_suite/                # Evaluation suite
├── mllm_tools/                # LLM interface wrappers
├── task_generator/            # Task and prompt generation
└── data/                      # Datasets
```

## 🚀 Quick Start

### 1. Requirements

- Python 3.8+
- FFmpeg (for video processing)
- LaTeX (for math formula rendering)
- Cairo and Pango (for Manim)

### 2. Installation

#### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    texlive-full \
    libcairo2-dev \
    libpango1.0-dev \
    libsdl-pango-dev \
    portaudio19-dev

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### macOS

```bash
# Install system dependencies
brew install ffmpeg
brew install cairo pango
brew install portaudio

# Install LaTeX
brew install --cask mactex

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
# Note: For macOS, uncomment pyobjc-related packages in requirements.txt
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy the environment template and configure your API keys:

```bash
cp .env.template .env
```

Edit the `.env` file with your model service credentials:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google Gemini
GOOGLE_API_KEY=your_google_api_key

# Moonshot Kimi
MOONSHOT_API_KEY=your_moonshot_api_key

# Custom API endpoint (optional)
CUSTOM_API_BASE=https://your-custom-endpoint.com
```

### 4. Set PYTHONPATH

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 5. Run Generation

#### Generate explanation for a single problem

```bash
python generate_explanation.py \
  --model "gpt-5" \
  --problem_path data/benchmark/benchmark.json \
  --output_dir output/my_experiment \
  --index 0 \
  --max_retries 3 \
  --translate_to_chinese
```

#### Batch generation for multiple problems

```bash
python generate_explanation.py \
  --model "claude-opus-4-6" \
  --problem_path data/benchmark/benchmark.json \
  --output_dir output/batch_experiment \
  --index 0,1,2,3,4 \
  --max_scene_concurrency 3 \
  --max_topic_concurrency 2 \
  --translate_to_chinese
```

### 6. View Results

Generated documents are located at:
```
output/my_experiment/<problem_name>/doc/
├── solution.md          # Explanation document
├── scene1.png          # Scene 1 diagram
├── scene2.png          # Scene 2 diagram
└── ...
```

## 🧰 Helper Scripts (`scripts/`)

Thin orchestration wrappers around `generate_explanation.py` / `evaluate.py` for whole-benchmark runs. Each script `cd`s to the repo root internally, so you can invoke it from anywhere as `bash scripts/<name>.sh`. They run detached and log under `output/<run>/`, and are **resumable** (re-running skips work already done).

### Full-benchmark generation — `scripts/run_<model>.sh`

`run_doubao_lite.sh`, `run_kimi_k27_code.sh`, `run_glm.sh`, `run_minimax3.sh` each generate the full 230-problem benchmark with one model. Models served by the local OpenAI-compatible **gateway** (doubao / glm / kimi / deepseek) are routed via a generated `.env.gateway` + `DOTENV_PATH` so they don't disturb the main `.env` (which points at the MiniMax endpoint); see the dotenv isolation note below.

```bash
bash scripts/run_kimi_k27_code.sh          # full 230 problems → output/kimi_k27_code/
bash scripts/run_kimi_k27_code.sh 0        # canary: only index 0 (verify image input / params first)
```

Output lands in `output/<model>/`; watch progress with `tail -f output/<model>/run_*.log`.

### Re-running failed problems — `scripts/rerun_*.sh`

`rerun_xelatex.sh` / `rerun_mhchem.sh` re-run the subset of problems that failed only because of a missing LaTeX dependency (once it's installed). They first delete the stale `scene*/render_failed.txt` and `scene*/code/scene*_code_tokens.json` markers — without that, the resume logic aborts the topic in ~0.01s instead of re-rendering. Use these as a template for "retry these specific indices after fixing the environment."

### Evaluation + report — `scripts/eval_eduillustrate.sh` (generic)

One generic script scores any generation run with an LLM judge and writes an edubenchmark-format report (`summary.json` / `scored.jsonl` / `report.html` + per-problem `evaluation_*.json`). It is **non-destructive** (new eval + report dirs; never deletes existing results).

```bash
bash scripts/eval_eduillustrate.sh <gen_dir> <gen_label> [judge_model] [data_path]

# Examples
bash scripts/eval_eduillustrate.sh output/minimax3      MiniMax-M3            # MiniMax-M3 self-judge
bash scripts/eval_eduillustrate.sh output/doubao_lite   doubao-seed-2.0-lite MiniMax-M3
bash scripts/eval_eduillustrate.sh output/kimi_k27_code kimi-k2.7-code       MiniMax-M3
```

- **Judge endpoint routing is automatic**: `MiniMax-M3`/`MiniMax-M2.7` judges use the main `.env` (MiniMax endpoint) as-is; any other judge is routed through the local gateway by temporarily rewriting `.env` (backed up and restored on exit) and setting `LITELLM_MAX_TOKENS=32768`.
- Eval results → `output/<gen>_eval_<judge>/`; report → `<edubenchmark>/reports/eval/eduillustrate/<label>__gen-full230_judge-<judge>/`.
- Tunable: `WORKERS=6 RETRY=3 bash scripts/eval_eduillustrate.sh ...`.

> **dotenv isolation:** both `litellm.py` and `evaluate.py` call `load_dotenv(override=True)`, so one process = one endpoint. To run a model on a different endpoint, the scripts either point `DOTENV_PATH` at an alternate env file (generation) or rewrite + restore `.env` (gateway judges). You cannot drive two endpoints from the same checkout in one process — but separate background runs on *different* endpoints (e.g. a gateway generation + a MiniMax evaluation) coexist fine.

## 📝 Usage Guide

### Command Line Arguments

#### Generation Parameters (`generate_explanation.py`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | LLM to use (e.g., gpt-5, claude-opus-4-6, Kimi-K25) | Required |
| `--problem_path` | Path to problem dataset JSON file | Required |
| `--output_dir` | Output directory | Required |
| `--index` | Problem index to process (single or comma-separated list) | - |
| `--max_retries` | Maximum retry attempts on errors | 3 |
| `--max_scene_concurrency` | Concurrent scenes within a single problem | 5 |
| `--max_topic_concurrency` | Concurrent problems to process | 1 |
| `--translate_to_chinese` | Translate output to Chinese | False |
| `--use_visual_fix_code` | Enable visual code fixing | False |
| `--disable_code` | Skip code generation (text only) | False |

#### Evaluation Parameters (`evaluate.py`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--eval_type` | Evaluation type: doc, explanation, text, image | Required |
| `--file_path` | Path to file or directory to evaluate | Required |
| `--output_folder` | Evaluation result output directory | Required |
| `--model_doc` | Model to use for document evaluation | gpt-5 |
| `--bulk_evaluate` | Batch evaluation mode | False |
| `--combine` | Combine all evaluation results | False |
| `--problem_data_path` | Original problem data path (for reference answers) | - |
| `--max_workers` | Number of concurrent evaluation processes | 4 |

### Workflow

EduIllustrate uses a multi-stage generation pipeline with two code generation strategies:

#### 1. Outline Planning

The system analyzes the problem and generates a structured outline, decomposing the explanation into text blocks `<TEXT_k>` and diagram scenes `<SCENE_k>`:

```xml
<SCENE_OUTLINE>
  <TEXT_1>First, let's understand the problem...</TEXT_1>
  <SCENE_1>Draw the geometric figure from the problem, annotate known conditions</SCENE_1>
  <TEXT_2>According to the Pythagorean theorem...</TEXT_2>
  <SCENE_2>Show the right triangle, highlight the relationship between three sides</SCENE_2>
  ...
</SCENE_OUTLINE>
```

#### 2. Code Generation Strategies

**Default Strategy (Incremental):**
- Generates a detailed implementation plan for Scene 1 only
- Scene 1 code is generated based on the implementation plan
- Subsequent scenes (Scene 2, 3, ...) are generated directly from:
  - The outline description for that scene
  - Scene 1's code as a reference example
- This approach maintains consistency by using Scene 1 as a style template

**All_Parallel Branch Strategy:**
- Generates detailed implementation plans for **all scenes** independently
- Each scene's code is generated based on its own implementation plan
- Scenes can be processed in parallel for faster generation
- Provides more flexibility but may have less style consistency

#### 3. Rendering

- Renders each scene using `manim -pql -s` (low quality + save last frame)
- Exports the last frame of each scene as a PNG image

#### 4. Document Assembly

- Assembles text blocks and scene images into a complete Markdown document
- Optional: Translates to Chinese (preserving all LaTeX formulas and formatting)

### Data Format

#### Input Data Format (JSON)

```json
[
  {
    "problem": "Problem description text...",
    "img": "base64-encoded problem image",
    "img_caption": "Image caption",
    "format_answer": "Standard answer",
    "topic": "physics",
    "grade": "9"
  }
]
```

#### Output Directory Structure

```
output/
└── my_experiment/
    └── problem_0_physics_g9/
        ├── doc/
        │   ├── solution.md        # Final explanation document
        │   ├── scene1.png        # Scene diagram
        │   └── scene2.png
        ├── scene1/
        │   ├── code/             # Manim code
        │   ├── media/            # Rendering output
        │   └── prompt.json       # Prompt records
        ├── scene2/
        │   └── ...
        └── timing.json           # Timing statistics
```

## 📊 Evaluation System

### Document Evaluation (--eval_type doc)

Evaluates the quality of generated illustrated explanation documents across 8 dimensions:

#### Text Dimensions (text only)

1. **Correctness and Completeness of Solution Steps** (0-5 points)
2. **Logical Coherence of Explanation** (0-5 points)
3. **Understandability and Teaching Effect** (0-5 points)
4. **Layout and Visual Clarity** (0-5 points)

#### Text-Diagram Synergy Dimensions

5. **Diagram Match with Problem** (0-5 points, each scene compared with original)
6. **Text-Diagram Synergy** (0-5 points, evaluates text-diagram coordination)

#### Diagram Dimensions

7. **Element Layout Quality** (0-5 points, each image evaluated independently)
8. **Visual Consistency** (0-5 points, all images compared with the first)

**Overall Score**: Geometric mean of all dimension scores

### Evaluation Examples

#### Evaluate a single document

```bash
python evaluate.py \
  --eval_type doc \
  --file_path "output/my_experiment/problem_0_physics_g9/doc" \
  --output_folder "output/doc_evaluation" \
  --model_doc "gpt-5" \
  --problem_data_path "data/benchmark/benchmark.json"
```

#### Batch evaluation

```bash
python evaluate.py \
  --eval_type doc \
  --file_path "output/my_experiment" \
  --output_folder "output/doc_evaluation" \
  --model_doc "claude-opus-4-6" \
  --bulk_evaluate \
  --combine \
  --max_workers 4
```

#### View evaluation results

```bash
# Individual problem evaluation result
cat output/doc_evaluation/evaluation_problem_0_physics_g9_*.json

# Combined summary results
cat output/doc_evaluation/combined_evaluation_*.json
```

Evaluation results include:
- Detailed scores and comments for each dimension
- Overall score
- Evaluation timestamp and model information
- Original problem reference information

## 🔧 Advanced Features

### 1. Visual Code Fixing

When enabled, the system uses rendered images as visual feedback to fix code errors:

```bash
python generate_explanation.py \
  --model "gpt-5" \
  --problem_path data/benchmark/benchmark.json \
  --output_dir output/visual_fix_test \
  --index 0 \
  --use_visual_fix_code
```

### 2. RAG Retrieval Augmentation

The system can use a vector database to retrieve similar examples to improve generation quality. After configuring an example codebase, the system automatically retrieves relevant references.

### 3. Custom Prompts

Modify prompt files in the `task_generator/prompts_raw/` directory, then regenerate:

```bash
cd task_generator
python parse_prompt.py
cd ..
```

### 4. Concurrency Optimization

For large-scale batch generation, optimize concurrency parameters:

```bash
python generate_explanation.py \
  --model "claude-opus-4-6" \
  --problem_path data/benchmark/benchmark.json \
  --output_dir output/large_batch \
  --max_scene_concurrency 5 \
  --max_topic_concurrency 3
```

- `max_scene_concurrency`: Number of scenes processed simultaneously within a single problem
- `max_topic_concurrency`: Number of problems processed simultaneously

## 🤝 Contributing

Issues and Pull Requests are welcome!

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd EduIllustrate

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📚 Citation

If this project helps your research, please cite:

```bibtex
@article{bi2026eduillustrate,
  title={EduIllustrate: Towards Scalable Automated Generation Of Multimodal Educational Content},
  author={Bi, Shuzhen and Zhang, Mingzi and Li, Zhuoxuan and Wang, Xiaolong and Li, Keqian and Zhou, Aimin and others},
  journal={arXiv e-prints},
  pages={arXiv--2604},
  year={2026}
}

```

## 🙏 Acknowledgments

This project is built upon these excellent open-source projects:

- [TheoremExplainAgent](https://github.com/TIGER-AI-Lab/TheoremExplainAgent) - Agent for theorem explanation video generation
- [Manim](https://github.com/ManimCommunity/manim) - Mathematical animation engine
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM API interface

## 📧 Contact

For questions or suggestions, please:

- Submit a GitHub Issue
- Email: bisz9918@gmail.com

---

**EduIllustrate** - Empowering Education with AI 🚀
