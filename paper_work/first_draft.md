# Do Safety Neurons Transfer? Localizing Refusal-Related MLP Neurons in GPT-2 Across NLI, Guardrail, and Instruction-Tuned Models

**50.040 Natural Language Processing, Fall 2026: Task 3 Custom Extension Proposal**
[Team name / members]

> **How to use this draft.** Body text is written to be submitted as is once the placeholders are filled. Lines in `[brackets]` are numbers or names to take from your repo. Blockquotes marked **Include** are notes for you and should be deleted before you submit. Target length: about 3.5 pages at 11pt.

---

## 1. Introduction

> **Include:** (1) the context, (2) the gap, (3) the research questions in italics, (4) contributions. Keep it to about 0.5 page.

Safety alignment is usually studied in billion-parameter chat models, where refusal behaviour has been traced to a single residual-stream direction (Arditi et al., 2024) and, more recently, to sparse sets of MLP neurons (Chen et al., 2025; Kazemi et al., 2026). Strikingly, Kazemi et al. (2026) report that refusal neurons already exist in *base* checkpoints before alignment, and that alignment mainly changes *when* they fire rather than *whether* they encode harm. If this holds, the neurons that detect harmful intent should be shared across models fine-tuned from the same base for very different purposes. These include a model trained only on natural language inference (NLI), a toxicity classifier, and a chat model.

GPT-2 (124M) is an ideal testbed for this question. In Tasks 1 and 2 we implemented it from scratch and fine-tuned it on English XNLI. The public ecosystem also contains several GPT-2 descendants post-trained with different data and objectives, and all of them start from the same pretrained weights. Because fine-tuning does not permute hidden units, neuron *i* in layer *l* refers to the same unit in every model. This makes cross-model intervention well defined.

We ask two research questions:

- *RQ1: In each model, is there a small set of MLP neurons whose activity is causally responsible for the safety-relevant decision (recognizing harmful intent, flagging toxicity, or refusing)?*
- *RQ2: Do these neurons transfer? Does suppressing or inducing the neurons found in one model change the safety behaviour of another model that was trained on a different dataset or task?*

Our contributions are:

- **Methodologically**, we reframe NLI as *intent recognition*. A model must decide whether a stylistically rewritten harmful prompt entails its plain-form request, and whether it contradicts a matched benign twin. This gives an NLI-trained model a safety-relevant task without any safety training.
- **Empirically**, we benchmark four instruction-tuned GPT-2 variants on SORRY-Bench and report their fulfillment rates under two judges.
- **Causally**, we localize candidate safety neurons in NLI, guardrail, and instruction-tuned GPT-2 models, then test whether steering them changes fulfillment rates both within and across models.

## 2. Related Work

> **Include:** three short paragraphs: refusal mechanisms, neuron-level localization (the method you adapt), and safety evaluation. End each with how your work differs. About 0.5 page.

**Refusal mechanisms.** Arditi et al. (2024) show that refusal in chat models is mediated by a single direction in the residual stream, computed as a difference in means between harmful and harmless prompts. Ablating this direction removes refusal. Geva et al. (2021) interpret transformer feed-forward layers as key-value memories. This motivates looking at individual MLP units instead of whole-layer directions.

**Safety neurons.** Chen et al. (2025) locate safety neurons by contrasting activations between an aligned and an unaligned version of the same model. They verify causality with dynamic activation patching and find that patching about 5% of neurons restores over 90% of safety performance. Kazemi et al. (2026) go further. They score each pre-down-projection MLP neuron by combining the gradient of a refusal log-odds loss with the activation difference between harmful and harmless prompts. Suppressing a *single* neuron gives about 91.7% attack success rate on JailbreakBench across Qwen3 and Llama-3.1 models. They also find these neurons in base checkpoints. Both works study models of 1.7B parameters or more that were explicitly safety-aligned. This work tests the same claims at 124M scale, and across *tasks* rather than only across alignment stages.

**Safety evaluation.** SORRY-Bench (Xie et al., 2025) offers class-balanced unsafe instructions across fine-grained categories, plus 20 linguistic mutations (e.g., question, slang, misspellings, role play) of each instruction. HarmBench (Mazeika et al., 2024) provides a fine-tuned Llama-2-13B classifier for judging whether a response fulfills a harmful request. We use the SORRY-Bench mutations as *premises* in our intent-NLI task, so the same data serves both the NLI and the generation evaluations.

## 3. Methods

> **Include:** one subsection per pipeline stage. Use a figure if space allows (pipeline: SORRY-Bench → twins → Intent-NLI → localization → steering → judge). About 1 page.

### 3.1 Models

> **Include:** this table. It answers the "need to write clearer about these models" point from your plan. Verify the training details marked † against each model card or paper before submitting.

All models share the GPT-2 small architecture (12 layers, d_model = 768, 3072 MLP units per layer, 36,864 MLP neurons in total) and are initialized from `openai-community/gpt2`.

| Role | Model | Post-training data | Objective | Safety data? |
|---|---|---|---|---|
| Task model | Ours: GPT-2 + XNLI-en (Task 2) | XNLI English, 392,702 pairs | Next-token prediction on label token | None |
| Instruction-tuned | `MBZUAI/LaMini-GPT-124M` | LaMini-instruction, about 2.58M instructions distilled from ChatGPT (Wu et al., 2024) | SFT | Not stated† |
| Instruction-tuned | `vicgalle/gpt2-alpaca-gpt4` | Alpaca-GPT4, 52k instructions with GPT-4 responses (Peng et al., 2023) | SFT | Not stated† |
| Instruction-tuned | `nicholasKluge/Aira-2-124M` | Instruct-Aira, about 124k synthetic prompt and completion pairs from ChatGPT, Llama, and Open-Assistant | SFT (5 epochs) | Not stated† |
| Preference-aligned | `nicholasKluge/Aira-2-124M-DPO` | Aira-2-124M + Reward-Aira (about 70k preference pairs labelled by Aira-RewardModel) | SFT → DPO | Not stated† |
| Guardrail | `uboza10300/finetuned-gpt2-hatexplainV2` | HateXplain (Mathew et al., 2021) | Sequence classification | Hate speech |
| Guardrail | `heegyu/gpt2-toxic-sequence-classification` | [unstated on card] | Sequence classification | Toxicity |
| Guardrail | `skrishna/gpt2-toxicity-classifier` | [unstated on card] | Sequence classification | Toxicity |

Notably, Aira-2-124M and Aira-2-124M-DPO form a matched *unaligned/aligned* pair. This is exactly the setting that Chen et al. (2025) require for activation contrasting.

### 3.2 Benign twin generation

For each SORRY-Bench base instruction, we generate a *benign twin* with [self-hosted LLM, name and size]. The twin keeps the surface form and register (including edgy wording) but changes the target or the action so that the request becomes harmless. A second variant changes both tone and topic. We manually check [N] twins for harmlessness and lexical overlap. This pairing controls for style, so a neuron that separates harmful prompts from their twins responds to *intent* rather than to keywords.

> **Include:** one example row (base / mutation / twin). Use a mild example, as in your plan.

### 3.3 NLI as intent recognition

We reuse the Task 2 prompt format, `Premise:{p} Hypothesis:{h} Label:`, without retraining:

- **Paraphrase pairs:** premise = a SORRY-Bench mutation (e.g., question or slang style); hypothesis = the base instruction. Expected label: *entailment (0)*.
- **Twin pairs:** premise = the same mutation; hypothesis = the benign twin. Expected label: *contradiction (1)* or *neutral (2)*.

A model that solves both conditions recognizes that a restyled request still carries the original intent. We report accuracy per condition and per mutation style.

### 3.4 Neuron localization

We adapt the scoring of Kazemi et al. (2026) to each model family. Let *a*ᵢ(*x*) be the post-GELU activation of MLP neuron *i* (the input to `c_proj`) at the decision token, and let *L*(*x*) be a model-specific decision score:

| Model family | Decision token | Decision score *L*(*x*) |
|---|---|---|
| NLI | `Label:` position | log p(entailment) − log p(contradiction) on harmful vs. twin pairs |
| Guardrail | last token | toxic-class logit − non-toxic logit |
| Instruction-tuned | last prompt token (end of template) | log-odds of a set of refusal phrases ("I'm sorry", "I cannot", …) |

Each neuron gets the score

  s_i = E_x[∂L/∂a_i] · ( E_harmful[a_i] − E_benign[a_i] ),

computed on [128] harmful and [128] twin prompts held out from evaluation. We keep the top-*k* neurons (k ∈ {1, 5, 10, 50, 0.1%, 1%}). For the Aira pair, we additionally compute Chen et al.'s change score (the activation difference between SFT and DPO models on the same prompts) as a second, gradient-free localization method.

### 3.5 Intervention

Following Kazemi et al. (2026), we apply two interventions at every token during prefill and generation:

- **Constant:** set *a*ᵢ to a fixed value *m*, chosen by sweep.
- **Anchor:** scale *a*ᵢ relative to its natural activation by factor *k* ∈ {−2, −1, 0, 1, 2}. This better preserves fluency.

*Suppression* (pushing neurons toward their benign-side value) should raise fulfillment. *Induction* (pushing toward the harmful-side value) should lower it. As controls, we steer *k* random neurons matched by layer and activation norm, and we track perplexity on [WikiText-2 / pretrain.txt] and XNLI test accuracy to measure capability cost.

### 3.6 Transfer protocol

For RQ2, neurons located in a *source* model (NLI, a guardrail, or one instruction-tuned model) are steered by index in each *target* instruction-tuned model. This gives a source × target transfer matrix of Δ fulfillment rate relative to the unsteered target.

## 4. Experimental Setup

> **Include:** datasets, judges, metrics in one compact paragraph, then subsections a–c. About 1 page.

**Data.** SORRY-Bench [version: 2024/06 release, 450 base instructions across 45 categories, or 2025/03 release, 440 across 44], with [list mutations used, e.g., question, slang, role play, technical terms, misspellings]. This yields [N] harmful prompts and [N] twins. The localization split ([256] prompts) is disjoint from the evaluation split.

**Judges.** HarmBench-Llama-2-13B-cls (Mazeika et al., 2024) and [Qwen model name and size, to confirm] as an LLM judge. We report agreement between the two judges (Cohen's κ) on [N] responses and manually inspect [N] disagreements.

**Metrics.** Intent-NLI accuracy; *fulfillment rate* (FR), the share of SORRY-Bench prompts whose response a judge labels as fulfilling the request; Δ FR under intervention; perplexity and XNLI accuracy as capability checks.

**Decoding.** Greedy, max [256] new tokens, each model's native prompt template (Alpaca-style for LaMini and Alpaca-GPT4; Aira special tokens for Aira-2).

### a) Initial results: NLI model on intent recognition

> **Include:** your numbers. One sentence of interpretation under the table. If twin accuracy is near chance, say so plainly; it motivates the guardrail fallback in §3.4.

| Model | Paraphrase pairs (expect E) | Twin pairs (expect C/N) | Overall |
|---|---|---|---|
| GPT-2 + XNLI-en (default hparams) | [acc] | [acc] | [acc] |
| GPT-2 + XNLI-en ([tuned hparams]) | [acc] | [acc] | [acc] |

Per-style breakdown (optional, if space allows): [table or one-line summary of the hardest mutation styles].

### b) Initial results: instruction-tuned models on SORRY-Bench

> **Include:** FR per model and per judge, plus the base GPT-2 as a reference. Point out whether DPO lowered FR compared with Aira-2 SFT, since this tells you whether a refusal signal exists to localize.

| Model | FR (HarmBench cls) | FR (Qwen judge) | κ |
|---|---|---|---|
| GPT-2 base (reference) | [ ] | [ ] | [ ] |
| LaMini-GPT-124M | [ ] | [ ] | [ ] |
| gpt2-alpaca-gpt4 | [ ] | [ ] | [ ] |
| Aira-2-124M | [ ] | [ ] | [ ] |
| Aira-2-124M-DPO | [ ] | [ ] | [ ] |

### c) Planned experiments and expected results

| # | Experiment | Expected outcome if the hypothesis holds | Outcome that would falsify it |
|---|---|---|---|
| E1 | Localize top-*k* neurons in each model (§3.4) | Sparse, high-scoring neurons concentrated in [mid] layers; significant index overlap across models | Scores near-uniform; overlap no better than random |
| E2 | Within-model steering (§3.5) | Suppression raises FR; induction lowers FR; random neurons show no effect | Random neurons have similar effect |
| E3 | Cross-model transfer matrix (§3.6) | NLI- and guardrail-derived neurons shift FR in instruction-tuned models | No transfer beyond the source model |
| E4 | Aira SFT vs. DPO change-score comparison | DPO reuses neurons already found in SFT (consistent with Kazemi et al., 2026) | DPO-specific neurons with no SFT counterpart |
| E5 | Capability cost | Anchor intervention keeps perplexity within [x]% and XNLI accuracy within [x] points | Large capability loss explains the FR change |

## 5. Discussion

> **Include:** what a positive or negative result would mean, the main risks, and next steps. About 0.5 page. Keep the limitations candid.

**Interpretation.** If neurons located in an NLI or toxicity model shift the fulfillment rate of chat models, this suggests that harmful-intent features are formed during pretraining and *reused* by downstream fine-tuning, rather than created by it. Simply put, safety at this scale would depend less on what post-training teaches and more on which pretrained units it connects to the output. A negative transfer result is still informative: it would indicate that each task builds its own decision circuit on top of shared lower-level features.

**Risks and limitations.**
- *Weak refusal baseline.* None of the four instruction-tuned models was trained on explicit refusal data, so baseline FR may already be near ceiling. In that case, raising FR by suppression cannot be measured, and *induction* (lowering FR) becomes the primary causal test.
- *Guardrail mismatch.* The three guardrails detect toxic or hateful *language*, whereas many SORRY-Bench categories (e.g., fraud, unqualified advice) are harmful in *intent* with neutral wording. We will report results per category to expose this gap, or post-train a GPT-2 classifier on [harmful/benign intent data] if needed.
- *Judge reliability.* Small models often produce incoherent text that judges may label inconsistently. We therefore report two judges and their agreement.
- *Scale.* Findings at 124M may not transfer to larger models. The value here is a controlled setting in which every model shares one initialization.

**Next steps.** (1) Use the localized neurons for alignment, for example by restricting a DPO update to the safety neurons or penalizing changes to them. (2) Fine-tune Aira-2-124M-DPO on new knowledge while freezing or regularizing the safety neurons, and test whether refusal is preserved, in line with known safety degradation under benign fine-tuning (Qi et al., 2024).

## Appendix: Proposed Task 3 Mark Breakdown (10 marks)

> **Include:** required by Section 5.2 of the project handout. Effort should match the default multilingual extension (2 + 2 + 3 + 3).

| Deliverable | Marks |
|---|---|
| Benign-twin generation and intent-NLI dataset construction (§3.2–3.3) | 1.5 |
| Intent-NLI evaluation of Task 2 models, per condition and per style (§4a) | 1.5 |
| SORRY-Bench evaluation of four instruction-tuned models with two judges (§4b) | 2 |
| Neuron localization across NLI, guardrail, and instruction-tuned models (E1, E4) | 2 |
| Within-model causal steering with random-neuron controls and capability checks (E2, E5) | 1.5 |
| Cross-model transfer matrix and analysis (E3) | 1.5 |
| **Total** | **10** |

## References

- Arditi, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. *NeurIPS 2024*.
- Chen, J., Wang, X., Yao, Z., Bai, Y., Hou, L., & Li, J. (2025). Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons. *NeurIPS 2025*. arXiv:2406.14144.
- Conneau, A., et al. (2018). XNLI: Evaluating Cross-lingual Sentence Representations. *EMNLP 2018*.
- Corrêa, N. K. (2023). Aira [model cards: nicholasKluge/Aira-2-124M, Aira-2-124M-DPO]. Hugging Face.
- Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2021*.
- Kazemi, H., Chegini, A., & Safi, M. (2026). A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models. arXiv:2605.08513.
- Mathew, B., et al. (2021). HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection. *AAAI 2021*.
- Mazeika, M., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. *ICML 2024*.
- Peng, B., et al. (2023). Instruction Tuning with GPT-4. arXiv:2304.03277.
- Qi, X., et al. (2024). Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! *ICLR 2024*.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.
- Rafailov, R., et al. (2023). Direct Preference Optimization. *NeurIPS 2023*.
- Wu, M., et al. (2024). LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions. *EACL 2024*.
- Xie, T., et al. (2025). SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal. *ICLR 2025*.