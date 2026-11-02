# Reproducing Meta-Harness (arXiv:2603.28052) with DeepSeek

FTEC5660 Reproducibility Work. This fork reproduces the **online text
classification** experiment of *Meta-Harness: End-to-End Optimization of Model
Harnesses* (Lee et al., 2026) and adds one controlled modification.

- Upstream: https://github.com/stanford-iris-lab/meta-harness (commit 8123cca)
- Paper: https://arxiv.org/abs/2603.28052

## What we did

1. **Reproduced the pipeline** on the vendored MCE datasets (USPTO,
   Symptom2Disease, LawBench) with the model swapped to DeepSeek
   (assignment "Edge case A" — see below).
2. **Reproduced the core claim**: a harness discovered by the agentic search
   (`straddle_coverage_bank`, iteration 3) beats both hand-designed baselines
   on validation (65.6% vs 62.2% few-shot-all / 44.2% zero-shot).
3. **Modification (isolated ablation)**: `straddle_arrival_order` replaces the
   discovered system's set-cover greedy *selection* with plain arrival order,
   keeping straddle clipping and the answer-vocabulary line identical.
   Result: 65.6% → 61.6% avg val (−4.0 points; Symptom2Disease 92.0% → 80.0%),
   showing the selection mechanism carries most of the gain.
4. **Test finalization**: one held-out test evaluation per frontier system via
   the repo's own protocol (`meta_harness.py --run-name repro-full-history
   --test`).

## Model swap (controlled variable, Edge case A)

| | Paper | This fork |
|---|---|---|
| Inner-loop classifier | GPT-OSS-120B (local vLLM, MXFP4) | `deepseek-chat` via `https://api.deepseek.com/v1` |
| temperature | 0.0 (config) | 0.0 (unchanged) |
| max_tokens | 16384 | 4096 (DeepSeek cap, see debug notes) |
| Proposer agent | Claude Code (Opus) | DeepSeek function-calling agent (`deepseek_proposer.py`) |

Absolute numbers are therefore **not directly comparable** to the paper's
Table 2; comparisons are made within our own setup (same tasks, metrics,
evaluation script, and model for every harness).

## Changes in this fork (see individual commits)

- `llm.py`: read API key from `LLM_API_KEY`/`DEEPSEEK_API_KEY`/`OPENAI_API_KEY`
  when an `api_base` is set; clamp `max_tokens` to 4096 for non-GPT-OSS models.
- `config.yaml`: model entry switched to `deepseek-chat` + DeepSeek base URL.
- `meta_harness.py`: **fixed an upstream bug** — `validate_candidates` ran the
  import check from `reference_examples/`, where `uv run` finds no project and
  every candidate failed with `ModuleNotFoundError: litellm`; now runs from the
  project directory with `PYTHONPATH=..`. Added `MH_PROPOSER` dispatch.
- `deepseek_proposer.py` (new): agentic DeepSeek proposer implementing the
  paper's proposer contract (filesystem access to prior code/scores/traces,
  writes candidates + `pending_eval.json`), with full session logging.
- `tests/test_evolution_contracts.py`: fixtures updated to the DeepSeek model
  name (28/28 pass).
- `agents/straddle_*.py` etc.: harnesses discovered by the search;
  `agents/straddle_arrival_order.py` is our ablation.
- `reproduction_results/`: evolution summary, frontier, and test results.

## Setup

```bash
cd reference_examples/text_classification
uv sync                                   # needs Python >= 3.11 (uv provides it)
export DEEPSEEK_API_KEY=sk-...            # never commit this
export MH_PROPOSER=deepseek               # use the DeepSeek proposer
```

Provider-free tests:

```bash
PYTHONPATH=.. uv run python -m unittest discover -s tests -v
```

## Reproduce

Baselines + 3 search iterations (≈15–20 min, a few million DeepSeek tokens):

```bash
uv run python meta_harness.py --iterations 3 --run-name repro-full-history
```

Ablation benchmark:

```bash
uv run python benchmark.py --logs-dir logs/repro-full-history \
  --results-dir logs/repro-full-history/results --memory straddle_arrival_order
```

One-shot held-out test evaluation (after selection is frozen):

```bash
uv run python meta_harness.py --run-name repro-full-history --test
```

## Results summary (validation; test in `reproduction_results/`)

| system | USPTO | S2D | LawBench | avg val |
|---|---|---|---|---|
| no_memory (zero-shot) | 26.7 | 76.0 | 30.0 | 44.2 |
| fewshot_all | 46.7 | 82.0 | 58.0 | 62.2 |
| **straddle_coverage_bank** (searched) | 46.7 | **92.0** | 58.0 | **65.6** |
| straddle_arrival_order (our ablation) | 46.7 | 80.0 | 58.0 | 61.6 |

Full per-candidate history: `reproduction_results/evolution_summary.jsonl`.
