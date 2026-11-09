# GPT-2 from scratch for NLI

Final project for 50.040 Natural Language Processing (Fall 2026). We build GPT-2 in PyTorch, fine-tune it for natural language inference (NLI) on XNLI, and then use the fine-tuned model to check whether a rewritten request still says the same thing as the original, using SORRY-Bench prompts (Task 3).

NLI is framed as next-token prediction. The prompt is `Premise: ...\nHypothesis: ...\nLabel:` and the model generates one token: `0` entailment, `1` contradictory, `2` neutral.

## Results

### Task 1: GPT-2 implementation
- Tokenizer, GPT-2 model and AdamW are done. `test_gpt2` matches the Hugging Face GPT-2, and the AdamW test passes against `optimizer_test.npy`.
- Toy pretraining (2 layers, hidden size 128, 3 epochs on `pretrain.txt`): average train loss 7.26, 6.86 and 6.61 after epochs 1, 2 and 3 (cumulative averages).

### Task 2: English NLI
Greedy generation works on both the pretrained and the toy model. Fine-tuned on 392,702 MultiNLI pairs, one epoch each, and evaluated on XNLI English (2,490 dev, 5,010 test):

| Run | LR | Batch size | Dev acc | Test acc |
|---|---|---|---|---|
| Baseline | 5e-5 | 4 | 78.15% | 76.63% |
| `lr2e-5` | 2e-5 | 4 | **80.12%** | **80.24%** |
| `bs16_lr1e-4` | 1e-4 | 16 | 78.47% | 78.64% |

- A lower learning rate gave the best accuracy and the lowest training loss.
- Batch size 16 gave a small gain and used 4x fewer steps (about 21 minutes against 79 minutes for `lr2e-5`).
- Dev and test accuracy move together, so the two sets look like the same distribution.
- Checkpoints are in `best_model/`, `best_model_lr2e-5/` and `best_model_bs16_lr1e-4/`.

### Task 3 (custom)
Uses SORRY-Bench: 450 unsafe base requests, each rewritten in 20 styles (question, role play, slang, translations, ciphers, persuasion styles, ...). We test the baseline model as is.

1. **Rewrite vs. base prompt.** Premise is the rewritten prompt, hypothesis is the base prompt. Expected label: entailment.
2. **Rewrite vs. benign twin.** For each base prompt, Qwen3.8-27B wrote a harmless twin with almost the same wording. Premise is the prompt in each style, hypothesis is the twin. Expected label: contradictory or neutral.

| Style | Entailment on rewrite vs. base | Non-entailment on rewrite vs. twin |
|---|---|---|
| `base` | n/a | 92.9% |
| `question` | 87.7% | 91.3% |
| `role_play` | 58.7% | 92.4% |
| `uncommon_dialects` | 54.2% | 87.1% |
| `slang` | 33.8% | 91.6% |
| persuasion styles (authority, expert, logic, evidence, misrepresentation) | 18-32% | 96-98% |
| ciphers and translations | 0-7% | 99-100% |

- The model recognises a rewrite as the same request mainly when the rewrite stays close to the original wording (`question`, `role_play`, `uncommon_dialects`). It is weak on heavier rewrites.
- For ciphers and translations the model cannot read the text and answers neutral or contradictory. These get 0-7% on the first test and 99-100% on the second, so only the readable styles tell us anything. `morse` prompts are all too long for the 1,023-token limit.
- Full per-style tables, with the split of predicted labels, are in `Task3/results/`.

## Repository

| Path | Contents |
|---|---|
| `default_final_project.ipynb` | Tasks 1 and 2: tokenizer, model, AdamW, pretraining, fine-tuning, hyperparameter runs |
| `model.py`, `config.py`, `utils.py` | GPT-2 model code, config and helpers (used by the Task 3 notebook) |
| `Task3/task3.ipynb` | Task 3 experiments |
| `Task3/data/` | `make_benign_twins.ipynb` (twin generation) and `sorry_bench_twins.jsonl` (the twins) |
| `Task3/results/` | `rewritten_vs_base.csv` and `rewritten_vs_twin.csv` |
| `best_model*/` | Checkpoints and training-progress files |
| `pretrain.txt`, `optimizer_test.npy` | Toy corpus and AdamW reference |

## Setup

Python 3.12, then:

```
pip install -r requirements.txt
```

XNLI is not in the repo. Download `XNLI-1.0.zip` and `XNLI-MT-1.0.zip` and extract them into the repo root, then run `default_final_project.ipynb` from top to bottom. Run the hyperparameter experiments before any other cell that replaces `train_dataset` or `test_dataset`.

**Task 3 notebook:** run it from inside `Task3/`. It finds `model.py` and `best_model/` in the repo root by itself. SORRY-Bench is a gated dataset: accept its terms on Hugging Face and set `HF_TOKEN` in the environment.

**Twin generation** (`Task3/data/make_benign_twins.ipynb`) runs Qwen3.8-27B with vLLM from the same `requirements.txt`; Qwen3.8 needs the newer `transformers` 5.x and `torch` 2.13. It needs a lot of memory: a GPU with at least 80 GB, and about 55 GB of disk for the model download.