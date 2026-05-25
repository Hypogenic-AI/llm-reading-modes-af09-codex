from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from sklearn.base import clone
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier


MODE_LABELS = [
    "human_original",
    "llm_generated",
    "llm_humanized",
    "human_polished",
    "dictated_spoken",
    "keyboard_noisy",
]

SEMEVAL_LABELS = ["human", "chatGPT", "cohere", "davinci", "bloomz", "dolly"]
DISFLUENCIES = {
    "um",
    "uh",
    "like",
    "you know",
    "kind of",
    "sort of",
    "i mean",
    "well",
}
KEYBOARD_NEIGHBORS = {
    "a": "sqwz",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdfr",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}


@dataclass
class Config:
    seed: int = 42
    custom_sample_size: int = 60
    semeval_per_class: int = 30
    generation_model: str = "gpt-4.1"
    classifier_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-large"
    max_generation_retries: int = 5
    max_classification_retries: int = 5
    bootstrap_iterations: int = 500
    permutation_iterations: int = 50
    group_folds: int = 5
    mode_temperature: float = 0.0
    batch_size_custom: int = 8
    batch_size_semeval: int = 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "results": root / "results",
        "figures": root / "figures",
        "logs": root / "logs",
        "cache": root / "results" / "cache",
        "cache_generations": root / "results" / "cache" / "generations",
        "cache_classification": root / "results" / "cache" / "classification",
        "cache_embeddings": root / "results" / "cache" / "embeddings",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_hc3(root: Path) -> pd.DataFrame:
    frames = []
    for domain in ["open_qa", "finance", "medicine"]:
        frame = pd.read_json(root / "datasets" / "hc3" / f"{domain}.jsonl", lines=True)
        frame["domain"] = domain
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["human_answer"] = df["human_answers"].apply(lambda xs: xs[0].strip() if xs else "")
    df["chatgpt_answer"] = df["chatgpt_answers"].apply(lambda xs: xs[0].strip() if xs else "")
    df["human_len_words"] = df["human_answer"].str.split().str.len()
    df = df[(df["human_len_words"] >= 60) & (df["human_len_words"] <= 260)].copy()
    df = df[df["question"].str.len() >= 10].copy()
    return df.reset_index(drop=True)


def sample_custom_anchors(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    per_domain = sample_size // 3
    rng = np.random.default_rng(seed)
    parts = []
    for domain in ["open_qa", "finance", "medicine"]:
        pool = df[df["domain"] == domain].copy()
        choose = min(per_domain, len(pool))
        indices = rng.choice(pool.index.to_numpy(), size=choose, replace=False)
        part = pool.loc[indices].copy()
        parts.append(part)
    sampled = pd.concat(parts, ignore_index=True)
    sampled["anchor_id"] = [f"hc3_{idx:03d}" for idx in range(len(sampled))]
    return sampled


def keyboard_noise(text: str, seed: int) -> str:
    rng = random.Random(seed)
    tokens = text.split()
    out = []
    for token in tokens:
        if len(token) < 4 or rng.random() > 0.18:
            out.append(token)
            continue
        chars = list(token)
        changed = False
        for idx, ch in enumerate(chars):
            base = ch.lower()
            if not base.isalpha():
                continue
            if rng.random() < 0.12:
                if base in KEYBOARD_NEIGHBORS and rng.random() < 0.7:
                    repl = rng.choice(list(KEYBOARD_NEIGHBORS[base]))
                    chars[idx] = repl.upper() if ch.isupper() else repl
                elif idx < len(chars) - 1 and chars[idx + 1].isalpha():
                    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                changed = True
                break
        if not changed and len(chars) > 5 and rng.random() < 0.5:
            drop_idx = rng.randrange(1, len(chars) - 1)
            del chars[drop_idx]
        out.append("".join(chars))
    noisy = " ".join(out)
    noisy = re.sub(r"\s+([,.;:!?])", r"\1", noisy)
    return noisy


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    payload = match.group(1) if match else cleaned
    return json.loads(payload)


def call_with_retry(fn, retries: int, sleep_s: float = 2.0):
    last_error = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            time.sleep(sleep_s * (attempt + 1))
    raise last_error


def response_to_usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def build_generation_prompt(question: str, human_answer: str) -> str:
    labels = {
        "llm_generated": "Answer the question directly in a polished, capable LLM style.",
        "llm_humanized": "Rewrite the llm_generated answer so it sounds more like an individual human response: slightly less uniform, more idiosyncratic, but still factually aligned.",
        "human_polished": "Polish the human answer for clarity and grammar while preserving meaning and evidence as much as possible.",
        "dictated_spoken": "Rewrite the human answer as if it came from live dictation or speech-to-text notes: spoken cadence, a few disfluencies, looser punctuation, but same core information.",
    }
    return (
        "You are creating controlled text variants for research on production modes.\n"
        "Keep semantic content close to the source materials.\n"
        "Return valid JSON with exactly these keys: "
        f"{list(labels.keys())}.\n\n"
        f"Question:\n{question}\n\n"
        f"Human answer:\n{human_answer}\n\n"
        "Instructions per field:\n"
        + "\n".join(f"- {key}: {value}" for key, value in labels.items())
        + "\n\nDo not include markdown fences."
    )


def generate_variants(
    client: OpenAI,
    anchor: pd.Series,
    config: Config,
    cache_dir: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    cache_path = cache_dir / f"{anchor.anchor_id}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        return payload["variants"], payload.get("usage", {})

    prompt = build_generation_prompt(anchor.question, anchor.human_answer)

    def make_call():
        return client.responses.create(
            model=config.generation_model,
            input=prompt,
            temperature=config.mode_temperature,
        )

    response = call_with_retry(make_call, config.max_generation_retries)
    parsed = extract_json(response.output_text)
    variants = {key: str(parsed[key]).strip() for key in parsed}
    payload = {
        "variants": variants,
        "usage": response_to_usage_dict(response),
        "anchor_id": anchor.anchor_id,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    return variants, payload["usage"]


def build_custom_dataset(
    client: OpenAI,
    anchors: pd.DataFrame,
    config: Config,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    usage_total = Counter()
    for _, anchor in anchors.iterrows():
        variants, usage = generate_variants(client, anchor, config, cache_dir)
        usage_total.update(usage)
        items = {
            "human_original": anchor.human_answer.strip(),
            "llm_generated": variants["llm_generated"].strip(),
            "llm_humanized": variants["llm_humanized"].strip(),
            "human_polished": variants["human_polished"].strip(),
            "dictated_spoken": variants["dictated_spoken"].strip(),
            "keyboard_noisy": keyboard_noise(anchor.human_answer.strip(), seed=hash(anchor.anchor_id) % (2**32)),
        }
        for mode, text in items.items():
            rows.append(
                {
                    "id": f"{anchor.anchor_id}::{mode}",
                    "anchor_id": anchor.anchor_id,
                    "domain": anchor.domain,
                    "question": anchor.question,
                    "mode": mode,
                    "text": text,
                }
            )
    df = pd.DataFrame(rows)
    df["word_count"] = df["text"].str.split().str.len()
    df["char_count"] = df["text"].str.len()
    return df, dict(usage_total)


def classify_batch(
    client: OpenAI,
    items: list[dict[str, str]],
    labels: list[str],
    model: str,
    retries: int,
    cache_dir: Path,
    task_name: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    batch_key = hashlib.sha256(
        json.dumps({"labels": labels, "items": items, "task": task_name}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = cache_dir / f"{task_name}_{batch_key}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        return payload["predictions"], payload.get("usage", {})

    prompt = (
        "You are labeling the production mode of texts.\n"
        f"Allowed labels: {labels}\n"
        "For each item, infer the most likely label from the text alone.\n"
        "Return JSON as a list of objects with keys: id, predicted_label, probabilities.\n"
        "probabilities must be a JSON object containing all labels with numeric values that sum to 1.\n"
        "No markdown fences.\n\n"
        f"Items:\n{json.dumps(items, ensure_ascii=False)}"
    )

    def make_call():
        return client.responses.create(model=model, input=prompt, temperature=0.0)

    response = call_with_retry(make_call, retries)
    parsed = extract_json(response.output_text)
    predictions = []
    for row in parsed:
        probs = {label: float(row["probabilities"][label]) for label in labels}
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        predictions.append(
            {
                "id": row["id"],
                "predicted_label": row["predicted_label"],
                "probabilities": probs,
            }
        )
    payload = {"predictions": predictions, "usage": response_to_usage_dict(response)}
    cache_path.write_text(json.dumps(payload, indent=2))
    return predictions, payload["usage"]


def run_batched_classification(
    client: OpenAI,
    df: pd.DataFrame,
    label_column: str,
    labels: list[str],
    batch_size: int,
    model: str,
    retries: int,
    cache_dir: Path,
    task_name: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    usage_total = Counter()
    all_predictions = []
    records = df[["id", "text"]].to_dict(orient="records")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        preds, usage = classify_batch(
            client=client,
            items=batch,
            labels=labels,
            model=model,
            retries=retries,
            cache_dir=cache_dir,
            task_name=task_name,
        )
        all_predictions.extend(preds)
        usage_total.update(usage)
    pred_df = pd.DataFrame(all_predictions)
    pred_df[label_column] = pred_df["predicted_label"]
    pred_df = pred_df.drop(columns=["predicted_label"])
    return pred_df, dict(usage_total)


def probability_matrix(df: pd.DataFrame, labels: list[str], probs_col: str) -> np.ndarray:
    return np.vstack(df[probs_col].apply(lambda row: [row[label] for label in labels]).to_numpy())


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    labels: list[str],
    iterations: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if metric_name == "accuracy":
            scores.append(accuracy_score(yt, yp))
        elif metric_name == "macro_f1":
            scores.append(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
        else:
            raise ValueError(metric_name)
    point = (
        accuracy_score(y_true, y_pred)
        if metric_name == "accuracy"
        else f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    return point, (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    labels: list[str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    acc_deltas = []
    f1_deltas = []
    n = len(y_true)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        acc_deltas.append(accuracy_score(yt, pa) - accuracy_score(yt, pb))
        f1_deltas.append(
            f1_score(yt, pa, labels=labels, average="macro", zero_division=0)
            - f1_score(yt, pb, labels=labels, average="macro", zero_division=0)
        )
    return {
        "accuracy_delta_mean": float(np.mean(acc_deltas)),
        "accuracy_delta_ci": [float(np.percentile(acc_deltas, 2.5)), float(np.percentile(acc_deltas, 97.5))],
        "macro_f1_delta_mean": float(np.mean(f1_deltas)),
        "macro_f1_delta_ci": [float(np.percentile(f1_deltas, 2.5)), float(np.percentile(f1_deltas, 97.5))],
        "accuracy_delta_p_two_sided": float(2 * min(np.mean(np.array(acc_deltas) <= 0), np.mean(np.array(acc_deltas) >= 0))),
        "macro_f1_delta_p_two_sided": float(2 * min(np.mean(np.array(f1_deltas) <= 0), np.mean(np.array(f1_deltas) >= 0))),
    }


def build_feature_vector(text: str) -> dict[str, float]:
    words = re.findall(r"\b\w+\b", text.lower())
    chars = len(text)
    word_count = len(words)
    sentences = max(1, len(re.split(r"[.!?]+", text.strip())) - 1)
    punctuation_count = len(re.findall(r"[,:;.!?\"'()\-]", text))
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / max(chars, 1)
    digit_ratio = sum(1 for ch in text if ch.isdigit()) / max(chars, 1)
    unique_ratio = len(set(words)) / max(word_count, 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    stop_ratio = sum(1 for w in words if w in ENGLISH_STOP_WORDS) / max(word_count, 1)
    disfluency_count = sum(text.lower().count(item) for item in DISFLUENCIES)
    typo_like = 0
    for word in words:
        if len(word) < 4:
            continue
        repeated = bool(re.search(r"(.)\1\1", word))
        no_vowel = not re.search(r"[aeiou]", word)
        mixed = bool(re.search(r"[a-z][0-9]|[0-9][a-z]", word))
        typo_like += int(repeated or no_vowel or mixed)
    return {
        "char_count": chars,
        "word_count": word_count,
        "avg_word_len": float(avg_word_len),
        "sentences": float(sentences),
        "punctuation_ratio": punctuation_count / max(chars, 1),
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "unique_ratio": unique_ratio,
        "stop_ratio": stop_ratio,
        "disfluency_per_word": disfluency_count / max(word_count, 1),
        "typo_like_per_word": typo_like / max(word_count, 1),
        "newline_ratio": text.count("\n") / max(chars, 1),
    }


def grouped_cv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator: Pipeline | LogisticRegression,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = len(np.unique(groups))
    splits = min(n_splits, unique_groups)
    cv = GroupKFold(n_splits=splits)
    preds = np.empty(len(y), dtype=int)
    probs = np.zeros((len(y), len(np.unique(y))), dtype=float)
    for train_idx, test_idx in cv.split(X, y, groups):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])
    return preds, probs


def embed_texts(
    client: OpenAI,
    texts: list[str],
    ids: list[str],
    model: str,
    cache_dir: Path,
    batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, int]]:
    usage_total = Counter()
    vectors = []
    uncached = []
    uncached_positions = []
    for pos, (text_id, text) in enumerate(zip(ids, texts)):
        cache_path = cache_dir / f"{text_id}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            vectors.append(np.array(payload["embedding"], dtype=float))
        else:
            vectors.append(None)
            uncached.append(text)
            uncached_positions.append((pos, text_id))

    for start in range(0, len(uncached), batch_size):
        batch_texts = uncached[start : start + batch_size]
        batch_positions = uncached_positions[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch_texts)
        usage_total.update(
            {
                "input_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                "output_tokens": 0,
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
            }
        )
        for item, (pos, text_id) in zip(response.data, batch_positions):
            vector = np.array(item.embedding, dtype=float)
            vectors[pos] = vector
            cache_path = cache_dir / f"{text_id}.json"
            cache_path.write_text(json.dumps({"id": text_id, "embedding": item.embedding}))
    matrix = np.vstack(vectors)
    return matrix, dict(usage_total)


def permutation_test_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator: Pipeline | LogisticRegression,
    observed_score: float,
    iterations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    unique_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(5, unique_groups))
    for _ in range(iterations):
        fold_preds = np.empty(len(y), dtype=int)
        permuted = rng.permutation(y)
        for train_idx, test_idx in cv.split(X, y, groups):
            model = clone(estimator)
            model.fit(X[train_idx], permuted[train_idx])
            fold_preds[test_idx] = model.predict(X[test_idx])
        scores.append(accuracy_score(y, fold_preds))
    scores = np.array(scores)
    return float((np.sum(scores >= observed_score) + 1) / (len(scores) + 1))


def summarize_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: list[str],
    probs: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if probs is not None:
        classes = labels
        true_idx = np.array([classes.index(label) for label in y_true])
        summary["nll"] = float(log_loss(true_idx, probs, labels=np.arange(len(classes))))
        one_hot = np.eye(len(classes))[true_idx]
        summary["brier"] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return summary


def save_confusion_figure(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_probe_figure(rows: list[dict[str, Any]], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    frame = pd.DataFrame(rows)
    sns.barplot(data=frame, x="model", y="accuracy", hue="model", palette="deep", legend=False)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("")
    plt.title("Grouped Mode Prediction Accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_length_figure(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="mode", y="word_count", hue="mode", palette="Set2", legend=False)
    plt.xticks(rotation=30, ha="right")
    plt.title("Word Count by Mode")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def delta_similarity_analysis(
    embed_df: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    rows = []
    base_vectors = {
        row.anchor_id: np.array(row.embedding, dtype=float)
        for row in embed_df[embed_df["mode"] == "human_original"].itertuples()
    }
    delta_map: dict[str, list[np.ndarray]] = {}
    for mode in labels:
        if mode == "human_original":
            continue
        vectors = []
        subset = embed_df[embed_df["mode"] == mode]
        for row in subset.itertuples():
            base = base_vectors[row.anchor_id]
            vectors.append(np.array(row.embedding, dtype=float) - base)
        delta_map[mode] = vectors

    rng = np.random.default_rng(42)
    for mode, vectors in delta_map.items():
        within = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                within.append(1 - cosine(vectors[i], vectors[j]))
        between = []
        other = [vec for other_mode, other_vecs in delta_map.items() if other_mode != mode for vec in other_vecs]
        sample_count = min(len(within), len(other))
        if sample_count == 0:
            continue
        sampled = rng.choice(len(other), size=sample_count, replace=False)
        ref = vectors[0]
        between = [1 - cosine(ref, other[idx]) for idx in sampled]
        rows.append(
            {
                "mode": mode,
                "within_mean": float(np.mean(within)),
                "within_std": float(np.std(within)),
                "between_mean": float(np.mean(between)),
                "between_std": float(np.std(between)),
                "within_n": len(within),
                "between_n": len(between),
                "within_values": within,
                "between_values": between,
            }
        )
    return pd.DataFrame(rows)


def test_delta_significance(delta_df: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    p_values = []
    interim = []
    for row in delta_df.itertuples():
        within = np.array(row.within_values)
        between = np.array(row.between_values)
        shapiro_p = min(
            shapiro(within[:5000]).pvalue if len(within) >= 3 else 0.0,
            shapiro(between[:5000]).pvalue if len(between) >= 3 else 0.0,
        )
        if shapiro_p > 0.05:
            stat = ttest_ind(within, between, equal_var=False, alternative="greater")
            test_name = "welch_t"
            p_value = float(stat.pvalue)
            statistic = float(stat.statistic)
        else:
            stat = mannwhitneyu(within, between, alternative="greater")
            test_name = "mann_whitney_u"
            p_value = float(stat.pvalue)
            statistic = float(stat.statistic)
        interim.append(
            {
                "mode": row.mode,
                "test": test_name,
                "statistic": statistic,
                "p_value_raw": p_value,
                "within_mean": float(np.mean(within)),
                "between_mean": float(np.mean(between)),
            }
        )
        p_values.append(p_value)
    order = np.argsort(p_values)
    adjusted = [None] * len(p_values)
    m = len(p_values)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        bh = min(prev, p_values[idx] * m / (m - rank + 1))
        adjusted[idx] = float(min(1.0, bh))
        prev = bh
    for item, adj in zip(interim, adjusted):
        item["p_value_bh"] = adj
        results.append(item)
    return results


def semeval_sample(root: Path, per_class: int, seed: int) -> pd.DataFrame:
    df = pd.read_json(root / "datasets" / "sem_eval_task8" / "subtaskB" / "subtaskB_dev.jsonl", lines=True)
    rng = np.random.default_rng(seed)
    parts = []
    for label in SEMEVAL_LABELS:
        pool = df[df["model"] == label].copy()
        choose = min(per_class, len(pool))
        idx = rng.choice(pool.index.to_numpy(), size=choose, replace=False)
        part = pool.loc[idx].copy()
        parts.append(part)
    sampled = pd.concat(parts, ignore_index=True)
    sampled["true_model"] = sampled["model"]
    sampled["id"] = sampled["id"].astype(str)
    return sampled


def environment_summary() -> dict[str, Any]:
    import matplotlib
    import openai
    import scipy
    import seaborn
    import sklearn
    import statsmodels

    return {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
        "statsmodels": statsmodels.__version__,
        "openai": openai.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=str, default=".")
    parser.add_argument("--custom-sample-size", type=int, default=60)
    parser.add_argument("--semeval-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--permutation-iterations", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.workspace_root).resolve()
    if str(root) != "/workspaces/llm-reading-modes-af09-codex":
        raise ValueError(f"Unexpected workspace root: {root}")

    config = Config(
        seed=args.seed,
        custom_sample_size=args.custom_sample_size,
        semeval_per_class=args.semeval_per_class,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
    )
    set_seed(config.seed)
    dirs = ensure_dirs(root)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    run_started = time.time()

    hc3 = load_hc3(root)
    anchors = sample_custom_anchors(hc3, config.custom_sample_size, config.seed)
    print(f"[stage] sampled {len(anchors)} HC3 anchors", flush=True)
    custom_df, gen_usage = build_custom_dataset(client, anchors, config, dirs["cache_generations"])
    print(f"[stage] built custom dataset with {len(custom_df)} rows", flush=True)
    custom_df.to_json(dirs["results"] / "custom_mode_dataset.jsonl", orient="records", lines=True)

    custom_preds, custom_cls_usage = run_batched_classification(
        client=client,
        df=custom_df,
        label_column="pred_mode",
        labels=MODE_LABELS,
        batch_size=config.batch_size_custom,
        model=config.classifier_model,
        retries=config.max_classification_retries,
        cache_dir=dirs["cache_classification"],
        task_name="custom_mode",
    )
    custom_eval = custom_df.merge(custom_preds, on="id", how="left")
    print("[stage] completed custom LLM classification", flush=True)
    custom_probs = probability_matrix(custom_eval, MODE_LABELS, "probabilities")
    custom_summary = summarize_metrics(custom_eval["mode"], custom_eval["pred_mode"], MODE_LABELS, custom_probs)
    custom_acc_point, custom_acc_ci = bootstrap_metric(
        custom_eval["mode"].to_numpy(),
        custom_eval["pred_mode"].to_numpy(),
        "accuracy",
        MODE_LABELS,
        config.bootstrap_iterations,
        config.seed,
    )
    custom_f1_point, custom_f1_ci = bootstrap_metric(
        custom_eval["mode"].to_numpy(),
        custom_eval["pred_mode"].to_numpy(),
        "macro_f1",
        MODE_LABELS,
        config.bootstrap_iterations,
        config.seed + 1,
    )
    custom_summary["accuracy_ci"] = list(custom_acc_ci)
    custom_summary["macro_f1_ci"] = list(custom_f1_ci)

    feature_df = pd.DataFrame([build_feature_vector(text) for text in custom_eval["text"]])
    X_feat = feature_df.to_numpy(dtype=float)
    label_encoder = LabelEncoder().fit(MODE_LABELS)
    y = label_encoder.transform(custom_eval["mode"])
    groups = custom_eval["anchor_id"].to_numpy()
    baseline_estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000, solver="liblinear"))),
        ]
    )
    feat_preds_idx, feat_probs = grouped_cv_predictions(X_feat, y, groups, baseline_estimator, config.group_folds, config.seed)
    feat_preds = label_encoder.inverse_transform(feat_preds_idx)
    feature_summary = summarize_metrics(custom_eval["mode"], feat_preds, MODE_LABELS, feat_probs)
    print("[stage] completed stylometric baseline", flush=True)

    print("[stage] requesting embeddings", flush=True)
    embeddings, embed_usage = embed_texts(
        client=client,
        texts=custom_eval["text"].tolist(),
        ids=custom_eval["id"].tolist(),
        model=config.embedding_model,
        cache_dir=dirs["cache_embeddings"],
    )
    embed_estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=config.seed)),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000, solver="liblinear"))),
        ]
    )
    embed_preds_idx, embed_probs = grouped_cv_predictions(embeddings, y, groups, embed_estimator, config.group_folds, config.seed)
    embed_preds = label_encoder.inverse_transform(embed_preds_idx)
    embedding_summary = summarize_metrics(custom_eval["mode"], embed_preds, MODE_LABELS, embed_probs)
    print("[stage] completed embedding probe", flush=True)
    print("[stage] running probe permutation test", flush=True)
    probe_perm_p = permutation_test_probe(
        embeddings,
        y,
        groups,
        embed_estimator,
        observed_score=embedding_summary["accuracy"],
        iterations=config.permutation_iterations,
        seed=config.seed,
    )
    embedding_summary["permutation_p"] = probe_perm_p

    paired_delta = paired_bootstrap_delta(
        custom_eval["mode"].to_numpy(),
        custom_eval["pred_mode"].to_numpy(),
        feat_preds,
        MODE_LABELS,
        config.bootstrap_iterations,
        config.seed,
    )

    embed_df = custom_eval[["id", "anchor_id", "mode"]].copy()
    embed_df["embedding"] = embeddings.tolist()
    delta_df = delta_similarity_analysis(embed_df, MODE_LABELS)
    delta_stats = test_delta_significance(delta_df)
    print("[stage] completed delta similarity analysis", flush=True)

    semeval_df = semeval_sample(root, config.semeval_per_class, config.seed)
    semeval_preds, semeval_usage = run_batched_classification(
        client=client,
        df=semeval_df,
        label_column="pred_model",
        labels=SEMEVAL_LABELS,
        batch_size=config.batch_size_semeval,
        model=config.classifier_model,
        retries=config.max_classification_retries,
        cache_dir=dirs["cache_classification"],
        task_name="semeval_subtaskB",
    )
    semeval_eval = semeval_df.merge(semeval_preds, on="id", how="left")
    print("[stage] completed SemEval classification", flush=True)
    semeval_probs = probability_matrix(semeval_eval, SEMEVAL_LABELS, "probabilities")
    semeval_summary = summarize_metrics(semeval_eval["true_model"], semeval_eval["pred_model"], SEMEVAL_LABELS, semeval_probs)
    semeval_acc_point, semeval_acc_ci = bootstrap_metric(
        semeval_eval["true_model"].to_numpy(),
        semeval_eval["pred_model"].to_numpy(),
        "accuracy",
        SEMEVAL_LABELS,
        config.bootstrap_iterations,
        config.seed + 3,
    )
    semeval_f1_point, semeval_f1_ci = bootstrap_metric(
        semeval_eval["true_model"].to_numpy(),
        semeval_eval["pred_model"].to_numpy(),
        "macro_f1",
        SEMEVAL_LABELS,
        config.bootstrap_iterations,
        config.seed + 4,
    )
    semeval_summary["accuracy_ci"] = list(semeval_acc_ci)
    semeval_summary["macro_f1_ci"] = list(semeval_f1_ci)

    save_confusion_figure(
        np.array(custom_summary["confusion_matrix"]),
        MODE_LABELS,
        dirs["figures"] / "custom_mode_confusion.png",
        "Custom Six-Mode Classification",
    )
    save_confusion_figure(
        np.array(semeval_summary["confusion_matrix"]),
        SEMEVAL_LABELS,
        dirs["figures"] / "semeval_subtaskB_confusion.png",
        "SemEval Subtask B Source Attribution",
    )
    save_probe_figure(
        [
            {"model": "Stylometric", "accuracy": feature_summary["accuracy"]},
            {"model": "Embedding probe", "accuracy": embedding_summary["accuracy"]},
            {"model": "LLM judge", "accuracy": custom_summary["accuracy"]},
        ],
        dirs["figures"] / "mode_probe_accuracy.png",
    )
    save_length_figure(custom_eval, dirs["figures"] / "custom_mode_lengths.png")

    delta_plot_df = delta_df[["mode", "within_mean", "between_mean"]].melt("mode", var_name="comparison", value_name="cosine")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=delta_plot_df, x="mode", y="cosine", hue="comparison", palette="muted")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean cosine similarity")
    plt.title("Transformation Vector Consistency")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "delta_similarity.png", dpi=200)
    plt.close()

    custom_eval.to_json(dirs["results"] / "custom_mode_predictions.jsonl", orient="records", lines=True)
    semeval_eval.to_json(dirs["results"] / "semeval_subtaskB_predictions.jsonl", orient="records", lines=True)
    feature_df.assign(true_mode=custom_eval["mode"], pred_mode=feat_preds).to_csv(
        dirs["results"] / "stylometric_features_and_predictions.csv", index=False
    )
    pd.DataFrame(delta_stats).to_json(dirs["results"] / "delta_similarity_stats.json", orient="records", indent=2)

    summary = {
        "config": asdict(config),
        "environment": environment_summary(),
        "gpu": {
            "detected": True,
            "model": "NVIDIA RTX A6000",
            "memory_total_mib": 49140,
            "memory_free_mib": 48541,
            "count": 4,
            "used_for_training": False,
            "notes": "This run used API-based LLM evaluation and CPU-based local analysis; GPUs were available but not required.",
        },
        "data_quality": {
            "hc3_candidates": int(len(hc3)),
            "anchors_sampled": int(len(anchors)),
            "custom_rows": int(len(custom_df)),
            "custom_mode_counts": custom_df["mode"].value_counts().to_dict(),
            "custom_domain_counts": custom_df.groupby("domain")["anchor_id"].nunique().to_dict(),
            "word_count_by_mode": custom_df.groupby("mode")["word_count"].agg(["mean", "std", "min", "max"]).round(2).to_dict(),
            "missing_values_custom": custom_df.isnull().sum().to_dict(),
        },
        "usage": {
            "generation": gen_usage,
            "custom_classification": custom_cls_usage,
            "embedding": embed_usage,
            "semeval_classification": semeval_usage,
        },
        "custom_mode_llm": custom_summary,
        "stylometric_baseline": feature_summary,
        "embedding_probe": embedding_summary,
        "llm_vs_stylometric_bootstrap": paired_delta,
        "semeval_subtaskB_llm": semeval_summary,
        "delta_similarity": delta_stats,
        "execution": {
            "started_unix": run_started,
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - run_started,
        },
    }
    (dirs["results"] / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(
        {
            "custom_accuracy": round(custom_acc_point, 4),
            "custom_macro_f1": round(custom_f1_point, 4),
            "stylometric_accuracy": round(feature_summary["accuracy"], 4),
            "embedding_probe_accuracy": round(embedding_summary["accuracy"], 4),
            "semeval_accuracy": round(semeval_acc_point, 4),
            "output_summary": str(dirs["results"] / "summary.json"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
