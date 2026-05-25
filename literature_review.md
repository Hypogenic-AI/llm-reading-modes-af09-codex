# Literature Review: "Modes" in LLM Reading

## Review Scope

### Research Question
What text "modes" can LLMs detect beyond simple human-vs-LLM authorship, and is "mode" a separable latent variable in text interpretation?

### Inclusion Criteria
- Focuses on detecting text origin, generation process, editing process, or mixed authorship
- Introduces or uses datasets for source attribution, boundary detection, or fine-grained authorship labels
- Includes behavioral or spoken/written signals relevant to non-semantic mode detection
- Provides code, data, or benchmark structure that can support downstream experiments

### Exclusion Criteria
- Pure watermarking or content provenance papers without usable text-side evaluation relevance
- Speech-only or keystroke-only papers with no link to text interpretation
- General LLM evaluation work without a mode-detection angle

### Time Frame
Primary focus: 2023-2025, with one targeted proxy paper for spoken-to-written conversion in 2024.

### Sources
- Manual arXiv search
- Official GitHub repositories
- Hugging Face datasets
- SemEval task repository and linked data

## Search Log

| Date | Query | Source | Notes |
|------|-------|--------|-------|
| 2026-05-25 | `machine-generated text detection`, `LLM-generated text detection` | arXiv/manual | Produced survey, SemEval, attribution, and fine-grained detector papers |
| 2026-05-25 | `keystroke dynamics LLM` | arXiv/manual | Produced keystroke-assisted-cheating literature |
| 2026-05-25 | `spoken to written ASR transcripts` | arXiv/manual | Produced CoS2W / SWAB benchmark paper |
| 2026-05-25 | `SemEval 2024 task 8 machine-generated text detection` | GitHub/manual | Produced official task repo and data links |

## Key Papers

### 1. A Survey on LLM-Generated Text Detection
- Authors: Wu et al.
- Year: 2023
- Source: arXiv
- Key contribution: Organizes detector families into watermarking, statistics-based, neural, and human-assisted methods.
- Methodology: Survey and synthesis.
- Datasets used: Reviews common detector benchmarks rather than introducing a new one.
- Results: Not an empirical benchmark paper; its value is in the taxonomy and failure modes.
- Code available: Yes, linked from the paper.
- Relevance: Best entry point for framing "mode" as a family of signals rather than a single binary label.

### 2. SemEval-2024 Task 8
- Authors: Wang et al.
- Year: 2024
- Source: SemEval / arXiv
- Key contribution: Establishes three related tasks: binary detection, generator attribution, and human-machine boundary detection.
- Methodology: Shared-task benchmark across domains, models, and languages with hidden surprise conditions.
- Datasets used: Extension of the M4 resource; task-specific train/dev/test sets.
- Baselines: Transformer encoder baselines for each subtask.
- Results:
  - Subtask B baseline accuracy: `74.61`
  - Best Subtask B accuracy: `90.85`
  - Subtask C baseline MAE: `21.54`
  - Best Subtask C MAE: `15.68`
- Code available: Yes, official repo cloned locally.
- Relevance: Strongest direct evidence that mode is not just binary authorship. Source identity and boundary position are both learnable.

### 3. LLM-DetectAIve
- Authors: Abassy et al.
- Year: 2024
- Source: EMNLP Demo / arXiv
- Key contribution: Expands machine-generated text detection into four classes: human-written, machine-generated, machine-humanized, and human-machine-polished.
- Methodology: Builds a large fine-grained dataset by taking M4GT-Bench style data and generating edited variants with multiple APIs and prompts.
- Datasets used: Derived from human text plus multi-provider LLM generation; total of `303,110` generated texts for the LLM-dependent classes.
- Baselines: RoBERTa, DeBERTa, DistilBERT.
- Results: Reports strong domain-specific performance, with RoBERTa around mid-`95%` macro-F1/accuracy on the shown domains.
- Code available: Yes, repo cloned locally.
- Relevance: Most directly aligned with the hypothesis that "mode" includes process labels like rewritten, polished, or humanized.

### 4. Detecting Machine-Generated Texts: Not Just "AI vs Humans"
- Authors: Ji et al.
- Year: 2024
- Source: arXiv
- Key contribution: Introduces an explicit "undecided" category and argues that explainability breaks under forced binary labeling.
- Methodology: Builds four datasets, identifies strong detectors and hard generators, then collects ternary labels plus explanation notes from annotators.
- Datasets used: New human/LLM datasets with a ternary follow-up set.
- Results: Main insight is conceptual: ambiguous cases are common enough that detection should represent them explicitly.
- Code available: Not found in gathered resources.
- Relevance: Important for latent-variable thinking. If mode is real, some texts should occupy intermediate regions instead of collapsing into binary labels.

### 5. TEXTMACHINA
- Authors: Sarvazyan et al.
- Year: 2024
- Source: arXiv
- Key contribution: Framework for generating datasets for detection, attribution, boundary detection, and mixcase tasks.
- Methodology: Human-text ingestion, prompt templating, constrained decoding, post-processing, bias mitigation, and task-specific generators.
- Datasets used: User-selected input corpora; examples use XSum and multiple providers.
- Results: Infrastructure contribution rather than benchmark leadership.
- Code available: Yes, repo cloned locally.
- Relevance: Likely the best practical tool here for building new mode datasets when public benchmarks are missing.

### 6. Source Attribution for LLM-Generated Data
- Authors: Wang et al.
- Year: 2023
- Source: arXiv
- Key contribution: Formalizes source attribution as identifying the data provider responsible for generated text, using watermarking.
- Methodology: Watermark-based attribution framework with robustness and accuracy goals.
- Datasets used: Synthetic texts generated with provider/source labels.
- Results: Shows that source-level attribution is technically feasible under their setup.
- Code available: Not gathered.
- Relevance: Expands the idea of mode from surface authorship to latent provenance.

### 7. Keystroke Dynamics Against Academic Dishonesty in the Age of LLMs
- Authors: Kundu et al.
- Year: 2024
- Source: IJCB / arXiv
- Key contribution: Detects AI-assisted writing from behavior rather than final text alone.
- Methodology: Modified TypeNet on keystroke sequences from genuine and assisted writing sessions across opinion-based and fact-based prompts.
- Datasets used: New keystroke dataset plus transfer tests with Buffalo and SBU keystroke corpora.
- Results:
  - Condition-specific accuracy: `74.98%` to `85.72%`
  - Condition-agnostic accuracy: `52.24%` to `80.54%`
- Code available: Not gathered.
- Relevance: Very important boundary case. It suggests some modes may be much easier to identify behaviorally than textually.

### 8. Recording for Eyes, Not Echoing to Ears
- Authors: Liu et al.
- Year: 2024
- Source: AAAI / arXiv
- Key contribution: Defines contextualized spoken-to-written conversion for ASR transcripts and introduces the SWAB benchmark.
- Methodology: Document-level conversion with context and auxiliary information, GPT-4 assisted construction, then human revision.
- Datasets used: SWAB, built from meeting, podcast, and lecture transcripts.
- Results: Shows LLMs can reliably recognize and rewrite spoken-mode artifacts into written-formal text.
- Code/data: Paper points to a public SWAB repository.
- Relevance: Best direct proxy for the "dictated/spoken vs written" mode hypothesis.

## Common Methodologies

- Fine-tuned encoder classifiers:
  - Used heavily in SemEval Task 8 and LLM-DetectAIve
  - RoBERTa and DeBERTa remain strong baselines

- Multi-class or structured mode labels:
  - Source attribution in SemEval Subtask B
  - Four-way authorship/process labels in LLM-DetectAIve
  - Boundary localization in SemEval Subtask C
  - Ternary explainability framing in Ji et al.

- Synthetic dataset generation:
  - Central in LLM-DetectAIve and TextMachina
  - Necessary when public corpora do not expose the desired mode variable

- Context-aware rewriting benchmarks:
  - CoS2W / SWAB models spoken-to-written style transfer as a mode change

- Behavioral features:
  - Keystroke dynamics are strong for AI-assistance detection, but they require data beyond final text

## Standard Baselines

- `RoBERTa` / `DeBERTa` fine-tuning for full-text classification
- `XLNet`, `Longformer`, `T5`, and ensemble variants in shared-task settings
- `TypeNet` for keystroke dynamics
- Zero-shot/statistical detectors such as entropy, rank, and Binoculars appear in the SemEval ecosystem but are generally weaker than tuned classifiers

## Evaluation Metrics

- Accuracy:
  - Standard for multi-class source attribution

- Macro-F1:
  - Important when label balance is imperfect or mode classes are heterogeneous

- MAE:
  - Used for boundary detection in mixed-mode text

- Calibration / abstention-aware analysis:
  - Not standardized yet, but the Ji et al. paper strongly suggests this should matter

- Correlation with human judgments:
  - Used in spoken-to-written evaluation when scoring formality and faithfulness

## Datasets in the Literature

- `M4` / `M4GT-Bench`:
  - Foundation for recent machine-generated text detection benchmarks

- SemEval-2024 Task 8:
  - Best gathered benchmark for binary detection, source attribution, and boundary detection

- LLM-DetectAIve dataset:
  - Best gathered example of fine-grained process labels

- SWAB:
  - Best discovered spoken-to-written benchmark for transcript mode

- HC3:
  - Useful binary human-vs-ChatGPT warm-up dataset, but less subtle than SemEval/LLM-DetectAIve

## Gaps and Opportunities

- There is strong literature for:
  - human vs machine
  - model/source attribution
  - mixed-authorship boundary detection
  - machine-humanized vs human-machine-polished text
  - spoken-to-written transcript style

- There is weaker literature for:
  - keyboard-layout inference from final text alone
  - typo-pattern-based latent mode discovery
  - jointly modeling multiple modes as disentangled latent variables

- The biggest open opportunity is a unified benchmark where the same underlying content is realized across multiple controlled modes:
  - human typed
  - human dictated
  - human typed with keyboard-layout perturbations
  - LLM generated
  - human text polished by LLM
  - LLM text humanized by editing

## Recommendations for Our Experiment

- Recommended datasets:
  - SemEval Subtask B for source attribution
  - SemEval Subtask C for mixed-mode boundary detection
  - HC3 for binary warm-up or sanity checks
  - LibriSpeech transcript sample only as a proxy while reconstructing a proper spoken-vs-written benchmark

- Recommended baselines:
  - RoBERTa-base and DeBERTa-v3 classifiers
  - A simple abstaining classifier or ternary scheme with an "uncertain" class
  - If available, a sequence labeling or regression baseline for span/boundary tasks

- Recommended metrics:
  - Macro-F1 and accuracy for document-level mode classification
  - MAE for boundary localization
  - Confidence calibration or abstention coverage for ambiguous modes

- Methodological considerations:
  - Avoid conflating topic/domain with mode
  - Hold content constant where possible and vary only the mode
  - Use surprise generators or unseen perturbations to test generalization
  - Include ambiguous/intermediate examples rather than forcing all samples into binary labels
  - Treat keyboard-layout and typo-pattern mode as likely requiring synthetic or newly collected data
