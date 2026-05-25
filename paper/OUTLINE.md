# Outline: Modes in LLM Reading

## Title
- LLMs Read Production Modes Better Than They Attribute Sources
- Subtitle idea: A Controlled Study of Mode Detection and Latent Mode Structure

## Abstract
- Motivate the shift from binary AI detection to broader production modes.
- State the gap: little work jointly studies matched-content multi-mode detection and latent mode structure.
- Summarize the controlled six-mode corpus built from HC3 anchors and OpenAI transformations.
- Report the main results: 1.000 custom accuracy for the LLM judge, 0.400 stylometric baseline, 0.489 grouped embedding probe, 0.256 external SemEval attribution accuracy.
- End with the narrower claim: broad process modes are readable; exact zero-shot generator identity is not.

## Introduction
- Hook: LLMs are now readers, judges, and filters, so their inferences about how text was produced matter.
- Importance: provenance, moderation, educational integrity, trust calibration.
- Gap: prior work covers binary detection, source attribution, mixed authorship, and spoken-to-written conversion separately.
- Approach: matched-content six-mode corpus plus LLM judgment, stylometric baseline, embedding probe, transformation-vector analysis, and external SemEval validation.
- Quantitative preview: 1.000 on custom six-way detection, 0.489 probe accuracy on unseen anchors, 0.256 on SemEval attribution.
- Contributions:
  - Build a controlled six-mode benchmark from HC3 anchors.
  - Show strong LLM sensitivity to broad production modes.
  - Provide latent-variable evidence from grouped probes and delta consistency.
  - Bound the claim with a negative external attribution result.

## Related Work
- Binary and source detection: survey and SemEval.
- Fine-grained process labels: LLM-DetectAIve, Ji et al., TextMachina.
- Spoken and behavioral signals: SWAB and keystroke dynamics.
- Positioning: unlike prior work, vary mode while roughly holding content fixed and test both surface classification and latent structure.

## Methodology
- Problem statement and hypotheses H1-H5 in prose.
- Controlled corpus construction:
  - 60 HC3 anchors.
  - 20 per domain across open_qa, finance, medicine.
  - Six modes per anchor.
- Models and settings:
  - GPT-4.1 generation and classification, text-embedding-3-large embeddings, temperature 0.
- Evaluations:
  - Zero-shot LLM judge.
  - Stylometric baseline with grouped splits.
  - Embedding probe: StandardScaler -> PCA(95\%) -> OvR logistic regression with GroupKFold.
  - Transformation-vector cosine consistency.
  - SemEval Subtask B external validation.
- Metrics and statistics:
  - Accuracy, macro-F1, bootstrap CIs, paired bootstrap deltas, permutation test, BH correction.

## Results
- Main results table comparing LLM judge, stylometric baseline, and embedding probe.
- Figure: perfect custom confusion matrix.
- Probe analysis:
  - 0.489 accuracy vs 0.167 chance.
  - Per-class recall table; dictated_spoken and keyboard_noisy strongest.
- Delta similarity table and figure:
  - All transformed modes significant; human_polished weakest.
- External validation table and confusion figure:
  - 0.256 accuracy, collapse toward human/chatGPT.
- Short ablation-style analysis:
  - Contrast overt modes vs subtle modes using recalls and delta magnitudes.

## Discussion
- Interpretation: hierarchy of detectable modes.
- Why the custom task is easy and why this does not imply robust provenance fingerprinting.
- Limitations:
  - same-family generation/classification
  - synthetic keyboard noise
  - approximate content matching
  - exploratory permutation test
  - zero-shot SemEval only
- Broader implications for screening, evaluation, and abstention-aware systems.

## Conclusion
- Restate the main answer: yes for broad mode detection and latent traces, no for zero-shot fine-grained source identity.
- Summarize contributions and practical lesson.
- Future work:
  - cross-family judge/generator
  - real dictated/typed and keyboard datasets
  - ambiguity-aware labels
  - transfer to mixed-authorship benchmarks
