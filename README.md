# nlptask2
NLP Project Collab with Jonny

Default final project for 50.040 Natural Language Processing, Fall 2026: GPT-2 built from scratch in PyTorch, then fine-tuned for natural language inference (NLI) on XNLI in English and other languages. All the work is in `default_final_project.ipynb`.

| Deadline | What's due |
|---|---|
| **16 Nov 2026, 23:59** | Build-it-break-it test cases (optional bonus, up to 20) |
| 19 Nov 2026 | Kaggle sign-up link released; one submission a day until the project deadline |
| **6 Dec 2026, 23:59** | Final project |

## Current status

**Task 1 is finished. Task 2 is written but has no results, because the baseline fine-tune was stopped at 1% on a CPU. Task 3 is partly scaffolded and has not been run.** Everything after the fine-tuning step depends on a `best_model/model.pt` checkpoint, which doesn't exist yet.

| Part | Marks | Code | Run | Result |
|---|---|---|---|---|
| **Task 1: GPT-2** | **10** | | | |
| Tokenizer | 2 | Done | Yes | Encodes and decodes correctly |
| GPT-2 model (TODO 1-5) | 2 | Done | Yes | Matches the Hugging Face GPT-2 (`test_gpt2` passes) |
| AdamW (TODO 6) | 2 | Done | Yes | Matches `optimizer_test.npy` (test passes) |
| Toy pretraining (TODO 7) | 4 | Done | Yes | Average loss falls from 10.86 to 6.61 |
| **Task 2: English NLI** | **10** | | | |
| Greedy generation (TODO 8) | 3 | Done | Yes | Works on both the pretrained and toy models |
| XNLI data loading | 2 | Done | Yes | 392,702 train / 2,490 dev / 5,010 test |
| Baseline fine-tune (TODO 9) | 5 (shared) | Done | **Stopped at 654 of 98,176 steps** | No accuracy yet |
| Test evaluation | shared | Done | No | None |
| Hyperparameter experiments (required) | shared | Done | No | None; observations still blank |
| **Task 3: Multilingual NLI** | **10** | | | |
| Zero-shot transfer, 15 languages | 2 | Provided | No | Needs `best_model/model.pt` |
| Fertility analysis | 2 | Provided | No | None |
| Per-language fine-tuning | 3 | **Not started** | No | Instructions only, no code |
| Unified multilingual fine-tuning | 3 | **Not started** | No | Instructions only, no code |
| **Build-it-break-it** | **3 bonus** | **Not started** | No | No team test cases in the repo yet |

The total is capped at 30 marks. Task 3 can be replaced by a custom extension, but that needs its own 10-mark rubric in the report. The notebook currently follows the default multilingual task.

## Results so far

**Toy pretraining** (2 layers, hidden size 128, 4 heads, 3 epochs on `pretrain.txt`). These are cumulative averages over the whole run, not per-epoch losses.

| After epoch | 1 | 2 | 3 |
|---|---|---|---|
| Average train loss | 7.26 | 6.86 | 6.61 |

**Text generation**, prompted with "Singapore University of Technology and Design (SUTD) is". Both models fall into repetition, which is common with greedy decoding.
- Pretrained GPT-2 is fluent at first, then loops on "full-time, full-time, ...".
- The toy model produces mostly noise, such as `"The "The "The`.

**Baseline fine-tune:** the only figure is an average loss of 1.11 at step 654. That is about the 1.10 expected from guessing among three labels, so it says nothing yet about final accuracy.

> The comment in the test evaluation cell mentions about 77.96% accuracy, but no saved output backs that up. It was already there in the first upload, so don't report it as our result.

## Next steps

1. **Write up to 20 build-it-break-it test cases by 16 Nov.** They must be English and use the same `Premise,Hypothesis,Label` columns as the example CSV. This doesn't depend on any training.
2. **Finish the baseline fine-tune on a GPU.** The saved run was on a CPU at 1.71 it/s, with about 16 hours left at the point it was stopped.
3. Run test evaluation, then the hyperparameter experiments, and fill in the observations.
4. Run zero-shot transfer and fertility analysis, then use them to choose languages.
5. Write the per-language and unified fine-tuning code.
6. Add a step that runs the chosen model on the build-it-break-it test set and writes a results CSV for Kaggle.

## Setup

Use Python 3.11 or newer (the notebook has been run on 3.12.9), then install the pinned packages:

```
pip install -r requirements.txt
```

XNLI isn't in the repo. Download `XNLI-1.0.zip` and `XNLI-MT-1.0.zip` and extract them into this folder, then run the notebook from top to bottom:

```
XNLI-1.0/xnli.dev.tsv
XNLI-1.0/xnli.test.tsv
XNLI-MT-1.0/multinli/multinli.train.{lang}.tsv
```

## Files

| File | Purpose |
|---|---|
| `default_final_project.ipynb` | All model, training and evaluation code |
| `xnli_eda.ipynb` | Short exploration of the XNLI splits, labels, text style and token lengths |
| `config.py` | `GPT2Config`, the model hyperparameters (the brief says no changes are needed) |
| `utils.py` | Attention mask and dtype helpers, GPT-2 size presets, and unused download code (the brief says no changes are needed) |
| `pretrain.txt` | Toy pretraining corpus, 5,778 lines |
| `optimizer_test.npy` | Reference weights for the AdamW test |
| `build_it_break_it - build_it_break_it.csv.csv` | 5 sample rows, which appear to be the course's format example for test cases; the notebook doesn't load it |
| `50040_nlp_default_project_2026Fall(1).pdf` | Project brief (not committed) |

## Things to watch out for

**Before re-running**
- **Reload the model before restarting the fine-tune.** The fine-tuning cell continues from whatever is in `model`. In a live kernel that includes the stopped run, so re-run the "Load a pretrained GPT-2" cell first.
- **Add mid-epoch checkpoints.** The model only saves at the end of an epoch, and the baseline has one epoch, so stopping early loses everything.
- **Run the hyperparameter experiments before Task 3.** Task 3 overwrites `test_dataset` and `test_loader` (ending on Chinese) and `train_dataset` (1% of Chinese). The experiments read those same globals, so running them afterwards would silently use the wrong data.
- **Run the experiment cells in order.** `baseline_results` and `finetune_and_evaluate` rely on variables set by earlier cells, such as `acc`, `best_dev_acc`, `dev_loader` and `test_loader`.

**Code issues**
- **`GPT2Model.from_pretrained` sets the wrong MLP size.** It uses `intermediate_size=d*3`, but GPT-2 uses `d*4`. Results aren't affected for now, because loading the real weights replaces the layer's tensors with the 3072-wide ones. However, `model.config` still reports 2304, and a checkpoint can't load into a model built with `d*3`.
- **`os` is only imported through `from utils import *`.** Add `import os` to the first cell, since the fine-tuning code needs it.
- **Activation differs slightly from GPT-2.** The notebook uses exact `F.gelu`, while GPT-2 uses the tanh approximation. The largest difference is about 0.0005 per activation, and `test_gpt2` is loose enough (`atol=0.1`) to pass anyway.
- **Checkpoint loads in Task 2 and Task 3 have no `map_location`.** A checkpoint saved on a GPU won't load on a CPU-only machine.
- **The Arabic train file has 2,000 empty hypotheses.** They are the last 2,000 rows, and `dropna` misses them because they are empty strings. Remove them before training on Arabic.
- **`XNLIDataset` can't read the build-it-break-it format.** That CSV uses capitalised columns and the label `contradiction`, while the class expects `premise`, `hypo` and `contradictory`. It will need a small loader of its own.

## Branches

Status as of the last fetch:
- `hai-branch` is the only branch with both the notebook and the `.py` and data files.
- `origin/main`, `Task-1`, `Task-3` and `Build-it-and-break-it` have had the notebook and README deleted, so merging needs care.
- `origin/amith/tokenizer-setup` has a tokenizer exploration section, setup files and a Windows UTF-8 fix that aren't on this branch yet.
