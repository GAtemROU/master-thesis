# Internal Verification Pipeline

## Overview
Project runs internal verification pipeline for math-style QA. Pipeline loads dataset, runs model prompts, extracts final answers, computes accuracy and verification confusion matrix, then writes run logs.

Main entry: `src/run_pipeline.py`.

## Pipeline flow
1. Load config (`src/iv_pipeline/config.py`) from JSON or defaults.
2. Load prompt templates (`src/iv_pipeline/prompt_templates/*`).
3. Build pipeline:
   - **VerificationPipeline**: constraints → baseline solution → constrained solution → verification.
   - **MajorityVotePipeline**: multiple sampled solutions → majority vote.
4. Load dataset (local file or Hugging Face for GSM8K).
5. Evaluate, compute metrics, write run log.

## Models
Defined in `src/iv_pipeline/models.py`:
- `mock_math`, `mock_constraints`, `mock_verifier` (default).
- `echo` (returns prompt).
- `hf_causal_lm` (Hugging Face causal LM).

Model objects cached by name + params.

## Prompts
Default prompt files in `src/iv_pipeline/prompt_templates/`:
- `task_prompt.txt`
- `constraint_prompt.txt`
- `verify_prompt.txt`

Override via config `prompts.task_prompt_path`, `prompts.constraint_prompt_path`, `prompts.verify_prompt_path`.

## Datasets
Implemented in `src/iv_pipeline/data.py`:
- `jsonl`: local JSONL with `question`, `answer` fields.
- `gsm8k`: local JSONL or Hugging Face via `hf:` spec.
- `math500`: local JSON or JSONL.

Hugging Face spec format for GSM8K:
```
hf:gsm8k?split=train[:1000]
```
If no split provided, default `train[:1000]`.

## Run logs and traces
Run logs written to `src/runs/run_<timestamp>.json` via `RunLogger`:
- inputs (args, config, prompts, dataset size)
- metrics
- details (optional per-example traces)
- events (verbose)
- status and errors

Trace files written to `src/traces/trace_<run_id>_idx<idx>.json` when `--trace-run`.

Use `src/show_latest_runs.py` to list recent runs.

## CLI usage
Run pipeline:
```
python src/run_pipeline.py [options]
```

Common options:
- `--dataset` path or `hf:` spec
- `--dataset-name` `jsonl|gsm8k|math500`
- `--config` path to config JSON
- `--pipeline` `proposed|majority_vote`
- `--num-samples` (majority vote)
- `--max-examples` cap dataset size
- `--trace-run` and `--trace-index`
- `--trace-only` to skip full evaluation
- `--log-traces` to store per-example outputs in run log
- `--verbose` for extra prints + run log events

## Config examples
`src/qwen3_gsm8k_hf.json`: Qwen3-1.7B on CUDA (float16).
`src/qwen3_gsm8k_hf_cpu.json`: Qwen3-1.7B on CPU (float32).

Both configure sampler, constraint, and verifier as `hf_causal_lm` with shared params.
