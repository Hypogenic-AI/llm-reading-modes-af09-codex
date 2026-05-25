# Modes in LLM Reading

## 1. Executive Summary
This project tested whether large language models can detect production "modes" in text beyond binary human-vs-AI authorship, and whether mode behaves like a latent variable in LLM-derived representations. Using a controlled six-mode corpus built from HC3 answers and real OpenAI transformations, `gpt-4.1` achieved perfect six-way classification on the custom benchmark, while a grouped stylometric baseline reached `0.400` accuracy and an embedding probe on `text-embedding-3-large` reached `0.489` accuracy on unseen content groups.

The strongest conclusion is not that LLMs can identify exact generator provenance. On an external six-class source-attribution benchmark from SemEval-2024 Task 8 Subtask B, the same `gpt-4.1` setup reached only `0.256` accuracy and mostly collapsed non-human systems into `chatGPT` or `human`. The evidence therefore supports a narrower claim: LLMs are very good at reading broad process modes such as dictated/spoken style, keyboard-noisy text, and LLM-post-edited prose, but much weaker at zero-shot fine-grained generator identity.

Practically, this suggests that mode detection is real and useful for provenance screening, editing-state inference, and ambiguity-aware evaluation, but current zero-shot prompting should not be trusted for source attribution across model families. The latent-variable evidence is promising but partial: mode is represented strongly enough to support grouped probing and stable transformation directions, yet the custom benchmark is easy enough that prompt-induced artifacts and same-family generation effects remain serious threats to validity.

## 2. Research Question & Motivation
The research question was: what text-production modes can current LLMs detect beyond simple human-vs-LLM authorship, and can "mode" be isolated as a latent variable in LLM interpretation?

This matters because LLMs are increasingly used as readers, judges, and filters. If they infer not only semantic content but also how text was produced, then moderation, educational-integrity tooling, provenance analysis, and downstream trust calibration all depend on understanding those inferences. The main gap in the gathered literature was the lack of a unified, content-controlled benchmark spanning multiple production modes rather than a single binary or attribution task.

## 3. Literature Review Summary
The local literature review identified four relevant clusters.

1. Binary and source detection.
   SemEval-2024 Task 8 and related detection work show that human-vs-machine classification and generator attribution are learnable in benchmark settings.
2. Mixed authorship and process labels.
   LLM-DetectAIve extends detection into labels such as machine-humanized and human-machine-polished text, directly motivating multi-mode evaluation.
3. Spoken-to-written conversion.
   SWAB and related work show that spoken/transcript artifacts are systematic enough for LLM rewriting and evaluation.
4. Behavioral and missing-mode signals.
   Keystroke-based work suggests production mode exists even beyond final text, while keyboard-layout and typo-pattern inference remain underexplored in text-only settings.

The main opportunity was a matched-content benchmark that varies mode while holding content roughly fixed, then tests whether mode remains visible to a reader model and to the reader model's embeddings.

## 4. Methodology

### 4.1 Experimental Setup
- Workspace root: `/workspaces/llm-reading-modes-af09-codex`
- Python: `3.12.8`
- Key libraries: `openai 2.38.0`, `scikit-learn 1.8.0`, `scipy 1.17.1`, `pandas 3.0.3`, `matplotlib 3.10.9`, `seaborn 0.13.2`
- GPU availability: 4 x `NVIDIA RTX A6000` (`49140 MiB` each, detected via `nvidia-smi`)
- GPU usage: none in the final pipeline because the study used API-based LLMs plus CPU-based statistics

### 4.2 Models and Parameters
- Generation model: `gpt-4.1`
- Classification model: `gpt-4.1`
- Embedding model: `text-embedding-3-large`
- Temperature: `0.0`
- Grouped evaluation: 5-fold `GroupKFold` by content anchor
- Embedding probe: `StandardScaler -> PCA(95% variance) -> OneVsRest(LogisticRegression)`
- Stylometric baseline: hand-crafted features with the same grouped evaluation protocol

### 4.3 Controlled Six-Mode Corpus
I sampled 60 HC3 items, balanced across `open_qa`, `finance`, and `medicine` with 20 anchors per domain. Each anchor contributed six matched modes:

- `human_original`
- `llm_generated`
- `llm_humanized`
- `human_polished`
- `dictated_spoken`
- `keyboard_noisy`

The first four transformed modes were produced with real `gpt-4.1` calls. `keyboard_noisy` was generated deterministically from the human answer using adjacent-key substitutions, transpositions, and omissions. The final custom dataset had 360 rows with no missing values.

### 4.4 External Validation
For out-of-distribution validation, I sampled 30 development examples from each SemEval-2024 Task 8 Subtask B class:

- `human`
- `chatGPT`
- `cohere`
- `davinci`
- `bloomz`
- `dolly`

Total external validation set size: 180 texts.

### 4.5 Metrics and Statistical Tests
- Accuracy and macro-F1 for all classification tasks
- Bootstrap confidence intervals for primary metrics
- Paired bootstrap comparison between the LLM judge and the stylometric baseline
- Embedding-probe permutation test against a shuffled-label null
- Within-mode vs between-mode cosine similarity tests for embedding transformation vectors
- Benjamini-Hochberg correction across per-mode transformation tests

### 4.6 API Usage
Recorded non-embedding usage from the final cached run:

| Stage | Input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|
| Controlled generation | 22,653 | 31,901 | 54,554 |
| Custom-mode classification | 58,457 | 38,175 | 96,632 |
| SemEval classification | 60,718 | 15,738 | 76,456 |
| Total recorded | 141,828 | 85,814 | 227,642 |

Embedding token usage was not preserved in the cache files and is therefore not reported exactly.

## 5. Results

### 5.1 Custom Six-Mode Classification

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| `gpt-4.1` zero-shot judge | `1.000` | `1.000` |
| Stylometric baseline | `0.400` | `0.386` |
| Embedding probe | `0.489` | `0.477` |

Bootstrap confidence intervals for the LLM judge on the custom benchmark:

- Accuracy 95% CI: `[1.000, 1.000]`
- Macro-F1 95% CI: `[1.000, 1.000]`

Paired bootstrap comparison, LLM judge minus stylometric baseline:

- Accuracy delta mean: `+0.602`
- Accuracy delta 95% CI: `[+0.555, +0.650]`
- Macro-F1 delta mean: `+0.618`
- Macro-F1 delta 95% CI: `[+0.573, +0.665]`

The confusion matrix was perfectly diagonal on the custom benchmark. Representative confidence examples were still moderate rather than saturated, for example:

- `human_original`: `0.85` probability on the correct class
- `human_polished`: `0.75`
- `llm_generated`: `0.85`
- `llm_humanized`: `0.80`
- `dictated_spoken`: `0.78`
- `keyboard_noisy`: `0.94`

This pattern suggests that the task was easy for the model but not solved only by outputting extreme probabilities.

### 5.2 Embedding Evidence for a Latent Mode Variable
The grouped embedding probe predicted mode at `0.489` accuracy on unseen content anchors. Since chance is `1/6 = 0.167`, this is materially above chance. A 20-run corrected permutation test produced null accuracies between `0.139` and `0.203`, with observed accuracy `0.489` and `p = 0.0476`.

Per-class recall in the embedding probe:

| Mode | Recall |
|---|---:|
| `dictated_spoken` | `0.700` |
| `keyboard_noisy` | `0.567` |
| `llm_humanized` | `0.567` |
| `llm_generated` | `0.617` |
| `human_original` | `0.250` |
| `human_polished` | `0.233` |

This supports the hypothesis that some modes are easier to separate than others. Overt production shifts such as spoken/transcript style and keyboard noise leave clearer latent traces than subtle polishing of human text.

### 5.3 Transformation-Vector Consistency
For each transformed mode, I measured the cosine similarity of embedding deltas from `human_original` to the transformed version and compared within-mode similarities against between-mode baselines.

| Mode | Within mean | Between mean | BH-adjusted p |
|---|---:|---:|---:|
| `keyboard_noisy` | `0.111` | `-0.015` | `< 1e-121` |
| `dictated_spoken` | `0.094` | `0.027` | `< 1e-60` |
| `llm_humanized` | `0.058` | `0.034` | `4.13e-13` |
| `llm_generated` | `0.036` | `0.014` | `6.67e-09` |
| `human_polished` | `0.029` | `0.021` | `0.0064` |

All five transformed modes showed statistically higher within-mode delta consistency than between-mode similarity. The weakest effect was `human_polished`, which matches the intuition that polishing is a subtler mode shift.

### 5.4 External Validation on SemEval Source Attribution

| Task | Accuracy | Macro-F1 | Chance |
|---|---:|---:|---:|
| SemEval Subtask B zero-shot attribution | `0.256` | `0.136` | `0.167` |

Bootstrap confidence intervals:

- Accuracy 95% CI: `[0.194, 0.311]`
- Macro-F1 95% CI: `[0.108, 0.164]`

The confusion matrix revealed a strong collapse:

- `human`: 29/30 correctly labeled
- `chatGPT`: 17/30 correctly labeled
- `cohere`, `davinci`, `bloomz`, `dolly`: almost always mapped to `chatGPT` or `human`

This is the most important negative result in the study. The same model that easily separated broad process modes did not recover fine-grained model identity zero-shot on an external benchmark.

### 5.5 Output Files
- Main summary: [results/summary.json](/workspaces/llm-reading-modes-af09-codex/results/summary.json)
- Controlled dataset: [results/custom_mode_dataset.jsonl](/workspaces/llm-reading-modes-af09-codex/results/custom_mode_dataset.jsonl)
- Controlled predictions: [results/custom_mode_predictions.jsonl](/workspaces/llm-reading-modes-af09-codex/results/custom_mode_predictions.jsonl)
- SemEval predictions: [results/semeval_subtaskB_predictions.jsonl](/workspaces/llm-reading-modes-af09-codex/results/semeval_subtaskB_predictions.jsonl)
- Feature baseline outputs: [results/stylometric_features_and_predictions.csv](/workspaces/llm-reading-modes-af09-codex/results/stylometric_features_and_predictions.csv)
- Figures: [figures/custom_mode_confusion.png](/workspaces/llm-reading-modes-af09-codex/figures/custom_mode_confusion.png), [figures/mode_probe_accuracy.png](/workspaces/llm-reading-modes-af09-codex/figures/mode_probe_accuracy.png), [figures/delta_similarity.png](/workspaces/llm-reading-modes-af09-codex/figures/delta_similarity.png), [figures/semeval_subtaskB_confusion.png](/workspaces/llm-reading-modes-af09-codex/figures/semeval_subtaskB_confusion.png), [figures/custom_mode_lengths.png](/workspaces/llm-reading-modes-af09-codex/figures/custom_mode_lengths.png)

## 6. Analysis & Discussion
The results support the main hypothesis in a qualified form.

First, broad production modes are clearly readable by a real LLM. In the controlled corpus, `gpt-4.1` cleanly distinguished original human answers, direct LLM generations, humanized LLM text, polished human text, dictated/spoken-style rewrites, and keyboard-noisy text. Because the stylometric baseline was much weaker, this does not reduce to a simple length-or-punctuation heuristic.

Second, the embedding analysis supports the latent-variable interpretation. The grouped probe succeeded on unseen content, and the mode-specific transformation vectors were more consistent within a mode than across modes. This is exactly what we would expect if "mode" behaves like a reusable representational direction rather than a purely example-specific surface label.

Third, the external benchmark sharply bounds the claim. The model did not demonstrate a strong zero-shot ability to separate `cohere`, `davinci`, `bloomz`, and `dolly` from `chatGPT`. That implies mode detection is strongest for high-level process properties such as editing state, speech-likeness, or noise patterns, not necessarily for robust fingerprinting of individual model families.

The most plausible interpretation is that LLMs detect a hierarchy of modes:

- Strongly detectable: keyboard noise, dictated/spoken cadence, broad human-vs-LLM process differences
- Moderately detectable: post-editing and humanization modes
- Weakly detectable zero-shot: exact generator identity among multiple non-human systems

## 7. Limitations
- The custom benchmark used `gpt-4.1` both to generate several transformed modes and to classify them. This creates a same-family recognition risk and may inflate accuracy.
- The perfect custom classification result is likely too strong to be taken as a calibrated real-world estimate. Prompt-induced artifacts may have made the classes cleaner than naturally occurring modes.
- `keyboard_noisy` was synthetic and deterministic. It captures a plausible typo mode, not a validated real keyboard-layout benchmark.
- The controlled corpus preserved content only approximately, not exactly. Some semantic drift may remain across transformed variants.
- The embedding permutation test was run with only 20 permutations after correcting the null, so the p-value should be treated as exploratory rather than definitive.
- SemEval evaluation was zero-shot only. Fine-tuned attribution models would almost certainly outperform the prompting baseline.
- The project did not run a span-level boundary experiment from SemEval Subtask C, so the latent-variable claim was tested at document level only.

## 8. Conclusions & Next Steps
The clearest answer to the research question is yes, LLMs do appear to detect multiple production modes beyond binary authorship, and those modes leave a measurable footprint in LLM-derived embedding space. However, the phenomenon is strongest for broad process states such as dictated/spoken style, keyboard noise, and post-editing, not for precise zero-shot source attribution among multiple generator families.

The next experiments should make the claim harder and cleaner:

1. Use one model family to generate data and a different family to judge it, to reduce self-recognition effects.
2. Collect real dictated-vs-typed and real keyboard-layout data instead of synthetic proxies.
3. Add abstention and ambiguity labels for borderline cases such as polished human text.
4. Extend the latent-variable analysis to SemEval Subtask C or another mixed-authorship benchmark.
5. Fit small supervised probes on external attribution datasets to test whether latent mode directions transfer beyond the custom corpus.

## 9. Reproducibility Notes
- Environment setup used the local `.venv` in this workspace.
- The main runner is [src/research_workspace/run_mode_experiments.py](/workspaces/llm-reading-modes-af09-codex/src/research_workspace/run_mode_experiments.py).
- Reproduction command:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m research_workspace.run_mode_experiments \
  --workspace-root /workspaces/llm-reading-modes-af09-codex \
  --custom-sample-size 60 \
  --semeval-per-class 30 \
  --bootstrap-iterations 200 \
  --permutation-iterations 20
```

API-backed stages are cached under `results/cache/` to make reruns faster and cheaper.

## 10. References
- Wu et al. 2023. *A Survey on LLM-Generated Text Detection.*
- Wang et al. 2024. *SemEval-2024 Task 8: Multigenerator, Multidomain, and Multilingual Black-Box Machine-Generated Text Detection.*
- Abassy et al. 2024. *LLM-DetectAIve.*
- Ji et al. 2024. *Detecting Machine-Generated Texts: Not Just "AI vs Humans".*
- Sarvazyan et al. 2024. *TEXTMACHINA.*
- Kundu et al. 2024. *Keystroke Dynamics Against Academic Dishonesty in the Age of LLMs.*
- Liu et al. 2024. *Recording for Eyes, Not Echoing to Ears.*
