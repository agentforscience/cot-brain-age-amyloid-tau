# Code

## Scripts written for this project

| File | Purpose |
|---|---|
| `arxiv_search.py` | Literature-search harness over the arXiv Atom API (fallback for the offline `paper-finder` service). Usage: `python code/arxiv_search.py <queries.json> <max_per_query> <out.json>` |
| `dl_papers.sh` | Bulk arXiv PDF downloader driven by `paper_search_results/selected.txt`; validates the `%PDF` magic bytes and retries. |
| `build_derived_datasets.py` | Builds the three analysis-ready tables in `datasets/derived/`. **Includes a from-scratch implementation of the Ding et al. 2025 L2C transform (eqs. 1–7).** |

## Cloned repositories

### `TAP-GPT/` — closest prior work ⭐
- **URL:** https://github.com/sophie-kearney/TAP-GPT
- **Paper:** `papers/2603.17191_Tabular_LLMs_interpretable_fewshot_AD_prediction.pdf` (IEEE JBHI 2026)
- **What it provides:** the reference implementation of few-shot tabular-LLM AD prediction —
  ICL-pool construction, tabular vs serialized prompt formats, interpretable prompting with
  structured outputs, missingness experiments, a multi-agent-system module, and TabPFN /
  traditional-ML task runners.
- **Key files:**
  - `core_functions/data_processing.py` — ICL pool + prompt table construction
  - `core_functions/models.py`, `core_functions/MAS.py` — model wrappers; multi-agent system
  - `interpretability/interpretable_TableGPT.py`, `interpretability/analyze_interpretable.py`
    — structured reasoning prompts and their analysis
  - `tasks/TabPFN_tasks.py`, `tasks/traditionalML_tasks.py`, `tasks/vanilla_llms_tasks.py`
    — baseline runners
  - `finetune/` — LoRA finetuning of TableGPT2 / Qwen2.5
- **Runnable here?** **No, not end-to-end.** It expects ADNI-derived tables
  (`ADNI_adnimerge_20170629_QT-freeze.csv`, QT-PAD) which are access-gated, and the finetuning
  path needs TableGPT2 weights. **Use it as a design reference**, not as an executable baseline.
  The prompt-construction and ICL-pool-disjointness patterns are the parts worth copying — the
  latter matters for avoiding leakage across splits.

### `TabPFN/` — tabular foundation model (baseline)
- **URL:** https://github.com/PriorLabs/TabPFN
- Installed as a package (`tabpfn` in `pyproject.toml`) and **verified working on GPU** with the
  locally staged checkpoints. The library's built-in weight download needs a license token that
  is unavailable here, so `model_path=` must be passed explicitly — see `models/README.md`.

### `tabpfn-extensions/` — interpretability + utilities for TabPFN
- **URL:** https://github.com/PriorLabs/tabpfn-extensions
- Contains SHAP-style interpretability helpers, unsupervised utilities and post-hoc ensembling.
  Relevant to constructing the **post-hoc-attribution arm** (the incumbent interpretability layer
  that CoT is being compared against).

### `TADPOLE/` — challenge reference code
- **URL:** https://github.com/noxtoby/TADPOLE
- Official evaluation code for the TADPOLE challenge: the **MAUC** and **BCA** implementations
  used by Ding et al. Worth reusing for metric fidelity even though the TADPOLE data itself is
  unavailable.

## Environment

Isolated venv created with `uv` at `.venv/`, pinned by `pyproject.toml`
(`[tool.uv] package = false` prevents `uv` from walking up to a parent project).

```bash
source .venv/bin/activate
```

Installed: `torch 2.13.0+cu130`, `transformers 5.15.1`, `accelerate`, `tabpfn`, `xgboost 3.4.1`,
`shap`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `huggingface-hub`,
`pypdf`, `httpx`, `requests`.

**Hardware verified:** 4 × NVIDIA RTX A6000 (49 GB each), 32 CPUs, 503 GB RAM, 774 GB free disk.
