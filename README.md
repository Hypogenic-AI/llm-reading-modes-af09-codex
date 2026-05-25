# Modes in LLM Reading

This project studies whether LLMs can detect text production "modes" such as direct LLM generation, human polishing, humanized LLM output, dictated/spoken style, and keyboard-noisy writing. The main result is a split finding: `gpt-4.1` separates those broad process modes extremely well in a controlled matched-content corpus, but performs poorly at zero-shot fine-grained generator attribution on SemEval-2024 Task 8 Subtask B.

Key findings:
- On a 360-text custom six-mode corpus, `gpt-4.1` reached `1.000` accuracy and `1.000` macro-F1.
- A grouped stylometric baseline reached `0.400` accuracy, and a grouped embedding probe reached `0.489` accuracy.
- Embedding-space mode prediction was above a corrected permutation null (`p = 0.0476` with 20 permutations), supporting a latent-variable interpretation.
- Transformation vectors were significantly more consistent within mode than across modes, strongest for `keyboard_noisy` and `dictated_spoken`.
- On SemEval Subtask B, zero-shot source attribution reached only `0.256` accuracy, with most non-human systems collapsed into `chatGPT` or `human`.

Full details are in [REPORT.md](/workspaces/llm-reading-modes-af09-codex/REPORT.md).

## Reproduce

```bash
source .venv/bin/activate
PYTHONPATH=src python -m research_workspace.run_mode_experiments \
  --workspace-root /workspaces/llm-reading-modes-af09-codex \
  --custom-sample-size 60 \
  --semeval-per-class 30 \
  --bootstrap-iterations 200 \
  --permutation-iterations 20
```

The run expects `OPENAI_API_KEY` in the environment. API outputs are cached under `results/cache/`.

## File Structure
- [REPORT.md](/workspaces/llm-reading-modes-af09-codex/REPORT.md): full report
- [planning.md](/workspaces/llm-reading-modes-af09-codex/planning.md): preregistered plan and novelty assessment
- [src/research_workspace/run_mode_experiments.py](/workspaces/llm-reading-modes-af09-codex/src/research_workspace/run_mode_experiments.py): end-to-end experiment runner
- [results/summary.json](/workspaces/llm-reading-modes-af09-codex/results/summary.json): machine-readable metrics
- [figures/](/workspaces/llm-reading-modes-af09-codex/figures): generated plots
