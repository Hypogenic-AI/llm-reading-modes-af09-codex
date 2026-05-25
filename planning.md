# Research Plan: "Modes" in LLM Reading

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are already used as judges, detectors, and assistants in settings where they implicitly interpret not just semantic content but also how text was produced. If models reliably infer production modes such as LLM-generation, post-editing, speech-like dictation, or keyboard-noise patterns, that affects moderation, provenance analysis, educational integrity, and how downstream systems should calibrate trust in text.

### Gap in Existing Work
The local literature review shows strong prior work on binary human-vs-machine detection, source attribution, mixed-authorship boundary detection, and spoken-to-written rewriting, but much weaker work on unified multi-mode detection under content controls and on treating "mode" as a latent variable rather than a task label. In particular, the gathered resources do not provide a benchmark that holds content approximately constant while varying production mode across multiple families.

### Our Novel Contribution
This project tests whether mode can be isolated as a cross-content latent variable by constructing a matched corpus where the same underlying answer is realized in multiple controlled modes. We then test two complementary claims: real LLMs can classify these modes from text alone, and LLM-derived embeddings encode stable mode directions that generalize across unseen content.

### Experiment Justification
- Experiment 1: Controlled six-way mode classification is needed to determine which production modes are directly detectable from text when content is approximately matched.
- Experiment 2: Embedding-space probing and transformation-vector analysis are needed to test the stronger latent-variable claim, not just raw classification accuracy.
- Experiment 3: SemEval Subtask B source-attribution evaluation is needed to verify that the observed phenomenon extends beyond our custom corpus to an external benchmark already identified in the literature.

## Research Question
What text production modes can current LLMs detect beyond simple human-vs-LLM authorship, and can "mode" be isolated as a latent variable in LLM interpretation of text?

## Background and Motivation
Recent benchmarks show that source attribution, machine-human boundary detection, and process labels such as machine-humanized or human-polished are learnable. What remains underexplored is whether these are separate tasks or instances of a broader mode-recognition ability. This project studies that question by combining a controlled multi-mode corpus with real-LLM judgments and embedding analysis.

## Hypothesis Decomposition
- H1: A real LLM can classify multiple production modes from text alone at accuracy materially above chance.
- H2: Some modes are easier than others; overtly surface-marked modes such as dictated/spoken-like text and keyboard-noisy text will be easier than subtle post-editing modes.
- H3: LLM embeddings contain cross-content information about mode, so a simple probe trained on embeddings will predict mode on unseen content above chance.
- H4: For each controlled transformation, the embedding delta from base text to transformed text will be more consistent within a mode than across different modes, suggesting a stable mode direction.
- H5: The same LLM will also perform above chance on external source-attribution data from SemEval Subtask B, showing that mode detection is not limited to the custom corpus.

Independent variables:
- Text mode: `human_original`, `llm_generated`, `llm_humanized`, `human_polished`, `dictated_spoken`, `keyboard_noisy`
- Dataset: controlled HC3-derived corpus vs. SemEval Subtask B sample
- Evaluation method: direct LLM classification, stylometric baseline, embedding probe

Dependent variables:
- Accuracy
- Macro-F1
- Confusion matrix structure
- Calibration metrics from model probabilities
- Probe accuracy under group-wise splits
- Transformation-vector cosine consistency

Alternative explanations to test against:
- Topic/domain leakage rather than mode
- Length differences rather than production mode
- Trivial lexical markers added by prompts
- Detection driven only by typos or punctuation sparsity

## Proposed Methodology

### Approach
Build a matched evaluation set from HC3 by sampling questions from multiple domains and treating each item as a content anchor. For each anchor, create six mode variants using a mix of original human text, real OpenAI generations, controlled post-editing prompts, and a deterministic keyboard-noise perturbation. Then evaluate:

1. zero-shot six-way mode classification with a real LLM
2. a shallow stylometric baseline for comparison
3. latent-variable analysis using OpenAI embeddings and simple probes
4. external source-attribution validation on SemEval Subtask B

This approach directly addresses the literature gap: content is approximately controlled, multiple mode families are present in one benchmark, and the latent-variable claim is tested separately from end-task accuracy.

### Experimental Steps
1. Sample approximately 60 HC3 items across `open_qa`, `finance`, and `medicine` with one human answer per item.
   Rationale: enough examples for statistical testing while keeping API cost manageable and domains heterogeneous.
2. Generate controlled variants for each item using a real OpenAI model.
   Rationale: required by the prompt and necessary to test real LLM behavior rather than simulated agents.
3. Add a deterministic keyboard-noise mode using adjacency and omission rules.
   Rationale: literature gap for text-only keyboard-related mode signals.
4. Query a real OpenAI model with a fixed classification prompt to predict one of the six modes and produce calibrated probabilities.
   Rationale: directly tests whether an LLM reads production mode from text alone.
5. Train a shallow baseline on hand-crafted stylometric features with grouped splits by content ID.
   Rationale: checks whether success is purely due to obvious surface cues.
6. Embed all texts with an OpenAI embedding model and train a grouped linear probe for mode prediction.
   Rationale: tests whether mode is encoded in representation space independent of anchor content.
7. Measure within-mode transformation-vector consistency from `human_original` to each transformed mode.
   Rationale: operationalizes the latent-variable hypothesis as a stable direction in embedding space.
8. Sample SemEval Subtask B dev data and run zero-shot source-attribution classification.
   Rationale: external validation using the strongest gathered benchmark for provenance-like mode detection.

### Baselines
- Chance and majority-class baselines
- Stylometric baseline using length, punctuation rate, digit ratio, lexical diversity, typo-like token ratio, disfluency marker counts, and sentence statistics
- For external validation, majority baseline and chance baseline for six-class source attribution

### Evaluation Metrics
- Accuracy and macro-F1 for all multi-class classification tasks
- Per-class precision and recall for subtle vs overt modes
- Brier score and negative log-likelihood if probability outputs are available
- Confusion matrices to identify nearby or ambiguous modes
- Grouped cross-validation accuracy for embedding probes
- Mean pairwise cosine similarity of transformation vectors within each mode
- Bootstrap confidence intervals for main metrics

### Statistical Analysis Plan
- Primary significance level: `alpha = 0.05`
- Use bootstrap confidence intervals for accuracy and macro-F1 on the controlled dataset
- Use paired bootstrap tests for comparing LLM classifier vs stylometric baseline on the same examples
- Use permutation testing for probe performance against chance under grouped labels
- Use Welch t-tests or Mann-Whitney tests, depending on normality, to compare within-mode delta similarity against cross-mode similarity
- Apply Benjamini-Hochberg correction across the family of per-mode similarity tests

## Expected Outcomes
Support for the hypothesis would look like:
- six-way mode classification materially above chance (`1/6 = 16.7%`)
- especially high recall for `dictated_spoken`, `keyboard_noisy`, and `llm_generated`
- lower but still above-chance performance on `llm_humanized` and `human_polished`
- grouped embedding probes above chance on unseen content
- transformation vectors that cluster by mode rather than behaving randomly
- above-chance source attribution on SemEval Subtask B

Evidence against the hypothesis would include near-chance classification once content is controlled, probe failure on grouped splits, or transformation vectors with no mode-specific consistency.

## Timeline and Milestones
1. Planning and resource verification: complete first
2. Environment and dependency completion: immediate next step
3. Data loading and controlled corpus construction: next
4. LLM evaluation and baseline modeling: next
5. Embedding and latent analysis: next
6. External SemEval validation: next
7. Figures, report, and reproducibility pass: final

## Potential Challenges
- API output formatting failures or transient request errors
  Mitigation: JSON schema prompt, retries, and response caching
- Generated variants drifting semantically away from the source
  Mitigation: prompt for content preservation and run spot checks with saved examples
- Keyboard-noise mode being trivially easy
  Mitigation: analyze it separately and avoid letting it dominate the broader conclusion
- Cost or runtime growth
  Mitigation: small pilot first, then scale to full sample after validation
- Domain confounds
  Mitigation: balanced sampling across HC3 domains and grouped evaluation by anchor content

## Success Criteria
- A complete reproducible pipeline in `src/` with cached model outputs in `results/`
- Actual experiments run with real OpenAI APIs
- At least one statistically supported result on controlled multi-mode detection
- A concrete answer to whether mode behaves like a latent variable in embedding space
- Final documentation in `REPORT.md` and `README.md` with results, limitations, and reproduction steps
