# Resources Catalog

## Summary

This catalog covers papers, datasets, and code repositories gathered for the project "Modes in LLM Reading". The most actionable resources are concentrated around three concrete mode families:

1. document-level authorship/source mode
2. span-level mixed-authorship mode
3. spoken-to-written and behavior-linked mode

## Papers

Total papers downloaded: `8`

| Title | Authors | Year | File | Key info |
|------|---------|------|------|----------|
| A Survey on LLM-Generated Text Detection | Wu et al. | 2023 | `papers/2310.14724_*.pdf` | Taxonomy of detector families and open problems |
| SemEval-2024 Task 8 | Wang et al. | 2024 | `papers/2404.14183_*.pdf` | Binary, attribution, and boundary benchmarks |
| LLM-DetectAIve | Abassy et al. | 2024 | `papers/2408.04284_*.pdf` | Four-way fine-grained authorship/process detection |
| Detecting Machine-Generated Texts: Not Just "AI vs Humans" | Ji et al. | 2024 | `papers/2406.18259_*.pdf` | Adds explicit ambiguity/undecided class |
| TEXTMACHINA | Sarvazyan et al. | 2024 | `papers/2401.03946_*.pdf` | Dataset generation framework |
| Source Attribution for LLM-Generated Data | Wang et al. | 2023 | `papers/2310.00646_*.pdf` | Provenance/source attribution framing |
| Keystroke Dynamics Against Academic Dishonesty in the Age of LLMs | Kundu et al. | 2024 | `papers/2406.15335_*.pdf` | Behavioral mode signal for AI assistance |
| Recording for Eyes, Not Echoing to Ears | Liu et al. | 2024 | `papers/2408.09688_*.pdf` | Spoken-to-written benchmark and task framing |

See [papers/README.md](/workspaces/llm-reading-modes-af09-codex/papers/README.md) for more detail.

## Datasets

Total dataset groups downloaded: `4`

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| SemEval-2024 Task 8 Subtask B | Google Drive via official repo | 71,027 train + 3,000 dev | source attribution | `datasets/sem_eval_task8/subtaskB/` | Best immediate dataset for multi-class mode/source detection |
| SemEval-2024 Task 8 Subtask C | Google Drive via official repo | 3,649 train + 505 dev | boundary detection | `datasets/sem_eval_task8/subtaskC/` | Best immediate dataset for mixed-mode span boundary detection |
| HC3 subsets | Hugging Face | 6,368 combined rows across downloaded subsets | binary human vs AI | `datasets/hc3/` | Good warm-up dataset |
| LibriSpeech ASR demo | Hugging Face | 73 examples | spoken transcript proxy | `datasets/librispeech_demo/` | Small proxy for spoken-text mode |

See [datasets/README.md](/workspaces/llm-reading-modes-af09-codex/datasets/README.md) for loading and download instructions.

## Code Repositories

Total repositories cloned: `3`

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| SemEval2024-task8 | `github.com/mbzuai-nlp/SemEval2024-task8` | official baselines and scorers | `code/SemEval2024-task8/` | Best baseline starting point |
| LLM-DetectAIve | `github.com/mbzuai-nlp/LLM-DetectAIve` | fine-grained detector | `code/LLM-DetectAIve/` | Useful four-way label schema |
| TextMachina | `github.com/Genaios/TextMachina` | synthetic dataset generation | `code/TextMachina/` | Useful for missing mode labels |

See [code/README.md](/workspaces/llm-reading-modes-af09-codex/code/README.md) for key files.

## Resource Gathering Notes

### Search Strategy

- Started with paper-oriented search around machine-generated text detection, source attribution, keystroke dynamics, and spoken-to-written conversion.
- Prioritized benchmark and tooling papers over isolated model papers.
- Preferred official task repositories and dataset hubs over third-party mirrors.

### Selection Criteria

- Direct relevance to "mode" as a detectable property of text or text production
- Availability of usable code or data
- Coverage of multiple levels:
  - binary origin
  - source identity
  - mixed authorship boundary
  - rewriting/polishing process
  - spoken/transcript style

### Challenges Encountered

- The local `paper-finder` helper did not return promptly, so the search was completed manually.
- Some recent datasets on Hugging Face still use old dataset-script layouts, which were less convenient than direct file downloads.
- No strong public benchmark was found for keyboard-layout inference from text-only error patterns.

### Gaps and Workarounds

- Gap: no gathered benchmark for dictated-vs-written final text in English that cleanly isolates the mode variable.
  - Workaround: use the spoken-to-written literature and a transcript proxy dataset now, then build or reconstruct a dedicated benchmark later.

- Gap: no gathered benchmark for keyboard-layout mode.
  - Workaround: use TextMachina or a custom perturbation pipeline to synthesize matched text under controlled typo/keyboard conditions.

## Recommendations for Experiment Design

1. Primary datasets:
   - Use SemEval Subtask B for multi-class source mode detection.
   - Use SemEval Subtask C for latent boundary or mixed-mode segmentation experiments.
   - Use HC3 for binary detector warm-up only.

2. Baseline methods:
   - Fine-tuned RoBERTa and DeBERTa encoders
   - A ternary or abstaining classifier to model ambiguity
   - Span localization model for mixed-mode boundary detection

3. Evaluation metrics:
   - Accuracy and macro-F1 for document-level classification
   - MAE for boundary localization
   - Confidence-aware analysis for ambiguous cases

4. Code to adapt/reuse:
   - SemEval baselines for immediate experiments
   - LLM-DetectAIve label framing for non-binary authorship modes
   - TextMachina for building missing datasets around keyboard-layout and typo-derived modes
