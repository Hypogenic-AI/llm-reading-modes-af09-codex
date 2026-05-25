# Cloned Repositories

## Repo 1: SemEval2024-task8
- URL: `https://github.com/mbzuai-nlp/SemEval2024-task8`
- Location: [code/SemEval2024-task8](/workspaces/llm-reading-modes-af09-codex/code/SemEval2024-task8)
- Purpose: Official shared-task code and documentation for binary detection, generator attribution, and boundary detection
- Key files:
  - [README.md](/workspaces/llm-reading-modes-af09-codex/code/SemEval2024-task8/README.md)
  - [subtaskA/baseline/transformer_baseline.py](/workspaces/llm-reading-modes-af09-codex/code/SemEval2024-task8/subtaskA/baseline/transformer_baseline.py)
  - [subtaskB/baseline/transformer_baseline.py](/workspaces/llm-reading-modes-af09-codex/code/SemEval2024-task8/subtaskB/baseline/transformer_baseline.py)
  - [subtaskC/baseline/transformer_baseline.py](/workspaces/llm-reading-modes-af09-codex/code/SemEval2024-task8/subtaskC/baseline/transformer_baseline.py)
  - scorers and format checkers in each subtask directory
- Notes:
  - The repo does not bundle train/dev data; it points to Google Drive downloads.
  - Useful as the most direct baseline implementation for the current project.

## Repo 2: LLM-DetectAIve
- URL: `https://github.com/mbzuai-nlp/LLM-DetectAIve`
- Location: [code/LLM-DetectAIve](/workspaces/llm-reading-modes-af09-codex/code/LLM-DetectAIve)
- Purpose: Four-way fine-grained detector for human-written, machine-generated, machine-humanized, and machine-polished text
- Key files:
  - [README.md](/workspaces/llm-reading-modes-af09-codex/code/LLM-DetectAIve/README.md)
  - [pipeline/main.py](/workspaces/llm-reading-modes-af09-codex/code/LLM-DetectAIve/pipeline/main.py)
  - [pipeline/model_pipeline.py](/workspaces/llm-reading-modes-af09-codex/code/LLM-DetectAIve/pipeline/model_pipeline.py)
  - [pipeline/dataset.py](/workspaces/llm-reading-modes-af09-codex/code/LLM-DetectAIve/pipeline/dataset.py)
  - [script/llm-detectaive.py](/workspaces/llm-reading-modes-af09-codex/code/LLM-DetectAIve/script/llm-detectaive.py)
- Notes:
  - Best direct example of extending "mode" labels beyond binary authorship.
  - Includes sample texts and a Hugging Face Space deployment path.

## Repo 3: TextMachina
- URL: `https://github.com/Genaios/TextMachina`
- Location: [code/TextMachina](/workspaces/llm-reading-modes-af09-codex/code/TextMachina)
- Purpose: Dataset generation framework for machine-generated text tasks
- Key files:
  - [README.md](/workspaces/llm-reading-modes-af09-codex/code/TextMachina/README.md)
  - [setup.py](/workspaces/llm-reading-modes-af09-codex/code/TextMachina/setup.py)
  - [text_machina/src/generators/detection.py](/workspaces/llm-reading-modes-af09-codex/code/TextMachina/text_machina/src/generators/detection.py)
  - [text_machina/src/generators/attribution.py](/workspaces/llm-reading-modes-af09-codex/code/TextMachina/text_machina/src/generators/attribution.py)
  - [text_machina/src/generators/boundary.py](/workspaces/llm-reading-modes-af09-codex/code/TextMachina/text_machina/src/generators/boundary.py)
  - [text_machina/src/generators/mixcase.py](/workspaces/llm-reading-modes-af09-codex/code/TextMachina/text_machina/src/generators/mixcase.py)
- Notes:
  - Strong tool if the next phase needs synthetic datasets for under-served modes like keyboard-layout or typo signatures.
  - Requires model/provider credentials for full generation workflows.
