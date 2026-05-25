# Downloaded Datasets

This directory contains local datasets for the reading-modes project. Data files are kept out of git by default via [datasets/.gitignore](/workspaces/llm-reading-modes-af09-codex/datasets/.gitignore).

## Dataset 1: SemEval-2024 Task 8 Subtask B

### Overview
- Source: `https://github.com/mbzuai-nlp/SemEval2024-task8`
- Local location: [datasets/sem_eval_task8/subtaskB](/workspaces/llm-reading-modes-af09-codex/datasets/sem_eval_task8/subtaskB)
- Size: `71,027` train + `3,000` dev examples, about `89 MB` including Subtask C
- Format: JSONL
- Task: multi-way authorship attribution among human, ChatGPT, Cohere, Davinci, BLOOMz, Dolly
- Splits: `subtaskB_train.jsonl`, `subtaskB_dev.jsonl`
- Sample files: `subtaskB_train_samples.json`, `subtaskB_dev_samples.json`

### Download Instructions

Using `gdown`:

```bash
source .venv/bin/activate
python -m gdown --folder https://drive.google.com/drive/folders/11YeloR2eTXcTzdwI04Z-M2QVvIeQAU6- -O datasets/sem_eval_task8/subtaskB
```

### Loading the Dataset

```python
import pandas as pd
train = pd.read_json("datasets/sem_eval_task8/subtaskB/subtaskB_train.jsonl", lines=True)
dev = pd.read_json("datasets/sem_eval_task8/subtaskB/subtaskB_dev.jsonl", lines=True)
```

### Notes
- Best ready-made dataset here for "which model produced this text?"
- Strong candidate for probing whether "mode" can be isolated beyond simple human-vs-LLM classification.

## Dataset 2: SemEval-2024 Task 8 Subtask C

### Overview
- Source: `https://github.com/mbzuai-nlp/SemEval2024-task8`
- Local location: [datasets/sem_eval_task8/subtaskC](/workspaces/llm-reading-modes-af09-codex/datasets/sem_eval_task8/subtaskC)
- Size: `3,649` train + `505` dev examples
- Format: JSONL
- Task: boundary detection in partially machine-generated text
- Splits: `subtaskC_train.jsonl`, `subtaskC_dev.jsonl`
- Sample files: `subtaskC_train_samples.json`, `subtaskC_dev_samples.json`

### Download Instructions

```bash
source .venv/bin/activate
python -m gdown --folder https://drive.google.com/drive/folders/16bRUuoeb_LxnCkcKM-ed6X6K5t_1C6mL -O datasets/sem_eval_task8/subtaskC
```

### Loading the Dataset

```python
import pandas as pd
train = pd.read_json("datasets/sem_eval_task8/subtaskC/subtaskC_train.jsonl", lines=True)
dev = pd.read_json("datasets/sem_eval_task8/subtaskC/subtaskC_dev.jsonl", lines=True)
```

### Notes
- Best ready-made dataset here for mixed-mode texts.
- Useful if "mode" is modeled as a latent boundary or span-level switch instead of a document-level label.

## Dataset 3: HC3 Domain Subsets

### Overview
- Source: `Hello-SimpleAI/HC3` on Hugging Face
- Local location: [datasets/hc3](/workspaces/llm-reading-modes-af09-codex/datasets/hc3)
- Downloaded subsets:
  - `finance.jsonl` with `3,933` rows
  - `open_qa.jsonl` with `1,187` rows
  - `medicine.jsonl` with `1,248` rows
- Format: JSONL
- Task: human vs ChatGPT-style responses
- Sample files: `finance_samples.json`, `open_qa_samples.json`, `medicine_samples.json`

### Download Instructions

```python
from huggingface_hub import hf_hub_download

hf_hub_download("Hello-SimpleAI/HC3", "finance.jsonl", repo_type="dataset", local_dir="datasets/hc3")
hf_hub_download("Hello-SimpleAI/HC3", "open_qa.jsonl", repo_type="dataset", local_dir="datasets/hc3")
hf_hub_download("Hello-SimpleAI/HC3", "medicine.jsonl", repo_type="dataset", local_dir="datasets/hc3")
```

### Loading the Dataset

```python
import json

with open("datasets/hc3/finance.jsonl") as f:
    rows = [json.loads(next(f)) for _ in range(10)]
```

### Notes
- Good lightweight binary pretraining or sanity-check dataset.
- Less useful for subtle mode distinctions than SemEval B/C.

## Dataset 4: LibriSpeech ASR Demo

### Overview
- Source: `hf-internal-testing/librispeech_asr_demo` on Hugging Face
- Local location: [datasets/librispeech_demo](/workspaces/llm-reading-modes-af09-codex/datasets/librispeech_demo)
- Size: `73` examples, about `8.8 MB`
- Format: Parquet
- Task role here: small spoken-transcript proxy dataset
- Files: `validation-00000-of-00001.parquet`, `samples.json`, `README.source.md`

### Download Instructions

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    "hf-internal-testing/librispeech_asr_demo",
    "clean/validation-00000-of-00001.parquet",
    repo_type="dataset",
    local_dir="datasets/librispeech_demo",
)
```

### Loading the Dataset

```python
import pandas as pd
df = pd.read_parquet("datasets/librispeech_demo/validation-00000-of-00001.parquet")
```

### Notes
- This is a proxy, not a dedicated dictated-vs-written benchmark.
- It is useful for building quick experiments around spoken transcript style, but a stronger next step is to reconstruct or obtain the SWAB benchmark from the CoS2W paper.

## Recommended Use

- Primary dataset for source mode: SemEval Subtask B
- Primary dataset for span-level/mixed mode: SemEval Subtask C
- Binary warm-up dataset: HC3
- Spoken-text proxy: LibriSpeech demo

## Gaps

- No public, ready-made text-only benchmark was found for keyboard-layout inference from naturalistic errors.
- The strongest paper for dictated/spoken mode uses the SWAB benchmark, but that benchmark was not directly packaged in the gathered resources.
- For the keyboard-layout and typo-pattern hypothesis, a synthetic-data path using TextMachina-style generation plus controlled typo injection remains likely necessary.
