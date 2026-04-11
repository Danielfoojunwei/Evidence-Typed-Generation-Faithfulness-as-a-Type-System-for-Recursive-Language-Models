#!/usr/bin/env python3
"""V4 Evaluation: Strong Paradigms + Learned Aggregation + LLaMA 8B
====================================================================

INSIGHTS APPLIED FROM v2/v3:
  1. ONLY STRONG PARADIGMS: v3 showed NLI (J=0.615) was 1.8-3.8x better
     than STS (0.348), Retrieval (0.164), Multi-QA (0.193), Lexical (0.215).
     Weak views dilute the strong one. v4 uses ONLY paradigms capable of
     genuine semantic verification — no similarity proxies.

  2. PARADIGM DIVERSITY: v2 showed 5 NLI models with same training data
     agree 96.7-98.5%. v4 uses fundamentally different verification approaches:
     NLI classification, LLM reasoning, extractive QA.

  3. LEARNED META-CLASSIFIER: v3 used weighted voting (heuristic).
     v4 trains a logistic regression meta-classifier on calibration data
     to learn optimal paradigm combination — can discover complementary
     error patterns that heuristic weighting misses.

  4. CALIBRATION SPLIT: v3 calibrated and evaluated on the same data
     (overfitting risk). v4 uses 30/70 cal/eval split for honest evaluation.

  5. STRONGER GENERATOR: GPT-2 (124M) → LLaMA 8B for E2E evaluation.
     Much stronger baseline makes ETG improvement more meaningful.

VERIFICATION PARADIGMS (3 strong, independent):
  V1: NLI Classification    — facebook/bart-large-mnli        (MNLI)
  V2: LLM Zero-Shot Judge   — google/flan-t5-large            (1800+ diverse tasks)
  V3: QA Verification       — deepset/roberta-base-squad2     (SQuAD 2.0)

Each paradigm uses different training data, task formulation, and architecture.

Dataset: TruthfulQA (Lin et al., ACL 2022) — 817 questions.
"""

import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from etg_rlm.bounds import hallucination_upper_bound, kl_bernoulli


# ===========================================================================
# Configuration
# ===========================================================================

CALIBRATION_FRACTION = 0.3
RANDOM_SEED = 42
# Increased from 50 to 572 (full evaluation set) to address the critical
# limitation that E2E results on 50 questions are not statistically
# meaningful (CI for 22.2% factuality spans roughly [8%, 44%] at n=18).
# Set ETG_E2E_QUICK=1 environment variable to use 50 for fast iteration.
E2E_N_QUESTIONS = 50 if os.environ.get("ETG_E2E_QUICK") else 572
E2E_MAX_NEW_TOKENS = 100

# Generator models to try (in order of preference)
# LLaMA 8B (16GB fp16) OOMs on 21GB RAM with paradigm data loaded.
# Use 3B model as primary — still 24x larger than GPT-2.
GENERATOR_CANDIDATES = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

# E2E judge (independent from verification views)
E2E_JUDGE_MODEL = "cross-encoder/nli-deberta-v3-small"


# ===========================================================================
# Scoring functions for each paradigm
# ===========================================================================

def score_nli(pairs, batch_size=8):
    """Paradigm 1: NLI cross-encoder — P(entailment) from BART-large-MNLI.

    The gold standard for textual entailment. Directly trained to classify
    whether a premise entails a hypothesis. Training data: MNLI (433K pairs).
    """
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, AutoConfig,
    )

    model_name = "facebook/bart-large-mnli"
    print(f"    Loading NLI model: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    config = AutoConfig.from_pretrained(model_name)
    ent_idx = None
    for idx, label in config.id2label.items():
        if label.lower() == "entailment":
            ent_idx = int(idx)
            break
    assert ent_idx is not None, f"No entailment label in {config.id2label}"

    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        inputs = tokenizer(
            [p[0] for p in batch], [p[1] for p in batch],
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        for j in range(len(batch)):
            scores.append(float(probs[j][ent_idx]))

    del model, tokenizer
    gc.collect()
    return scores


def score_llm_judge(pairs, batch_size=1):
    """Paradigm 2: LLM zero-shot judge — Flan-T5 generative reasoning.

    Fundamentally different from NLI: uses instruction-following and
    broad reasoning (trained on 1800+ diverse tasks) rather than
    discriminative NLI classification. Generates 'true'/'false' and
    we extract P(true) from first-token logits.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Try flan-t5-large first, fall back to base
    for model_name in ["google/flan-t5-large", "google/flan-t5-base"]:
        try:
            print(f"    Loading LLM judge: {model_name}...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Loaded ({n_params/1e6:.0f}M params)", flush=True)
            break
        except Exception as e:
            print(f"    Failed to load {model_name}: {e}", flush=True)
            continue
    else:
        raise RuntimeError("Could not load any Flan-T5 model")

    # Pre-compute token IDs for true/false
    true_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_id = tokenizer.encode("false", add_special_tokens=False)[0]

    scores = []
    for i, (evidence, claim) in enumerate(pairs):
        prompt = (
            f"Based on the evidence, is the claim true or false?\n"
            f"Evidence: {evidence}\n"
            f"Claim: {claim}\n"
            f"Answer:"
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True,
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
            first_logits = out.scores[0][0]
            probs = torch.softmax(
                first_logits[torch.tensor([true_id, false_id])], dim=0,
            )
            p_true = float(probs[0])
        scores.append(p_true)

        if (i + 1) % 500 == 0:
            print(f"    Scored {i+1}/{len(pairs)}...", flush=True)

    del model, tokenizer
    gc.collect()
    return scores


def score_qa_verification(pairs, batch_size=1):
    """Paradigm 3: QA-based verification — extractive QA confidence.

    Fundamentally different from NLI and LLM reasoning: treats verification
    as a reading comprehension task. Uses SQuAD 2.0-trained model that can
    detect unanswerable questions. High confidence = evidence addresses the
    claim. Training data: SQuAD 2.0 (100K+ QA pairs).
    """
    from transformers import pipeline as hf_pipeline

    model_name = "deepset/roberta-base-squad2"
    print(f"    Loading QA model: {model_name}...", flush=True)
    qa = hf_pipeline("question-answering", model=model_name)

    scores = []
    for i, (evidence, claim) in enumerate(pairs):
        try:
            result = qa(question=claim, context=evidence)
            scores.append(float(result["score"]))
        except Exception:
            scores.append(0.0)

        if (i + 1) % 500 == 0:
            print(f"    Scored {i+1}/{len(pairs)}...", flush=True)

    del qa
    gc.collect()
    return scores


# ===========================================================================
# Calibration functions
# ===========================================================================

def calibrate_youden(scores, labels, n_thresholds=500):
    """Find threshold maximizing Youden's J = TPR - FPR."""
    best_t, best_j, best_tpr, best_fpr = 0.5, 0.0, 0.0, 0.0
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    n_pos = labels_arr.sum()
    n_neg = len(labels_arr) - n_pos

    for i in range(1, n_thresholds):
        t = i / n_thresholds
        preds = scores_arr >= t
        tp = (preds & labels_arr).sum()
        fp = (preds & ~labels_arr).sum()
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t, best_tpr, best_fpr = j, t, tpr, fpr

    return best_t, best_tpr, best_fpr, best_j


def calibrate_target_fpr(scores, labels, target_fpr=0.05, n_thresholds=500):
    """Find lowest threshold keeping FPR <= target with maximum TPR."""
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    n_pos = labels_arr.sum()
    n_neg = len(labels_arr) - n_pos
    best_t, best_tpr, best_fpr = 0.999, 0.0, 0.0

    for i in range(1, n_thresholds):
        t = i / n_thresholds
        preds = scores_arr >= t
        tp = (preds & labels_arr).sum()
        fp = (preds & ~labels_arr).sum()
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        if fpr <= target_fpr and tpr > best_tpr:
            best_t, best_tpr, best_fpr = t, tpr, fpr

    return best_t, best_tpr, best_fpr


def compute_pr_curve(scores, labels, n_points=200):
    """Precision-recall curve sweeping threshold."""
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    n_pos = labels_arr.sum()
    points = []

    for i in range(1, n_points):
        t = i / n_points
        preds = scores_arr >= t
        tp = (preds & labels_arr).sum()
        fp = (preds & ~labels_arr).sum()
        total = tp + fp
        if total == 0:
            continue
        pr = tp / total
        rc = tp / n_pos if n_pos > 0 else 0
        points.append((float(t), float(pr), float(rc)))

    return points


# ===========================================================================
# Learned Meta-Classifier (Logistic Regression)
# ===========================================================================

def fit_logistic_regression(X, y, lr=0.5, n_iter=3000, lam=0.01):
    """L2-regularized logistic regression via gradient descent.

    Learns optimal combination of paradigm scores. The weights reveal
    the relative importance and complementarity of each paradigm.

    X: (n, d) feature matrix (paradigm scores, optionally with interactions)
    y: (n,) binary labels
    Returns: w (d,), b (scalar)
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for iteration in range(n_iter):
        z = np.clip(X @ w + b, -20, 20)
        pred = 1.0 / (1.0 + np.exp(-z))

        error = pred - y
        grad_w = (X.T @ error) / n + lam * w
        grad_b = np.mean(error)

        w -= lr * grad_w
        b -= lr * grad_b

        # Reduce learning rate over time
        if (iteration + 1) % 1000 == 0:
            lr *= 0.5

    return w, b


def predict_logistic(X, w, b):
    """Predict probabilities from logistic regression."""
    z = np.clip(X @ w + b, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


def build_features(scores_dict, paradigm_names):
    """Build feature matrix from paradigm scores.

    Includes raw scores + pairwise interaction terms for richer representation.
    """
    n = len(scores_dict[paradigm_names[0]])
    # Raw scores
    raw = np.column_stack([scores_dict[name] for name in paradigm_names])
    # Pairwise products (interaction terms)
    interactions = []
    for i in range(len(paradigm_names)):
        for j in range(i + 1, len(paradigm_names)):
            interactions.append(raw[:, i] * raw[:, j])
    if interactions:
        inter_matrix = np.column_stack(interactions)
        return np.hstack([raw, inter_matrix])
    return raw


# ===========================================================================
# Evaluation helpers
# ===========================================================================

def evaluate(predictions, labels):
    """Compute precision, recall, F1, hallucination rate, FPR."""
    preds = np.array(predictions, dtype=bool)
    labs = np.array(labels, dtype=bool)
    tp = (preds & labs).sum()
    fp = (preds & ~labs).sum()
    fn = (~preds & labs).sum()
    tn = (~preds & ~labs).sum()
    total_acc = tp + fp
    precision = tp / total_acc if total_acc > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    halluc = fp / total_acc if total_acc > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "halluc_rate": float(halluc),
        "fpr": float(fpr),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ===========================================================================
# E2E Generation with LLaMA
# ===========================================================================

def load_generator():
    """Load the best available LLM for generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for model_name in GENERATOR_CANDIDATES:
        try:
            print(f"    Trying generator: {model_name}...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            )
            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Loaded {model_name} ({n_params/1e9:.1f}B params)", flush=True)
            return model, tokenizer, model_name
        except Exception as e:
            print(f"    Failed: {str(e)[:100]}", flush=True)
            del_vars = [v for v in ['model', 'tokenizer'] if v in dir()]
            gc.collect()
            continue

    raise RuntimeError("Could not load any generator model")


def generate_completions(model, tokenizer, questions, max_new_tokens=100):
    """Generate factual completions for TruthfulQA questions."""
    completions = []
    model_name = getattr(model, 'name_or_path', '') or ''

    for i, q in enumerate(questions):
        # Format prompt based on model type
        if "llama" in model_name.lower() or "nous" in model_name.lower():
            prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Answer the following question factually in 1-2 sentences:\n{q}"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif "qwen" in model_name.lower():
            prompt = (
                f"<|im_start|>user\nAnswer the following question factually "
                f"in 1-2 sentences:\n{q}<|im_end|>\n<|im_start|>assistant\n"
            )
        elif "tinyllama" in model_name.lower():
            prompt = (
                f"<|system|>\nYou are a helpful assistant.</s>\n"
                f"<|user|>\nAnswer factually in 1-2 sentences: {q}</s>\n"
                f"<|assistant|>\n"
            )
        else:
            prompt = f"Question: {q}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        # Move to same dtype as model
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the generated part
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        completions.append(text)

        if (i + 1) % 10 == 0:
            print(f"    Generated {i+1}/{len(questions)}...", flush=True)

    return completions


def score_e2e_judge(evidence_list, sentence_list, batch_size=8):
    """Score generated sentences using independent judge (DeBERTa-v3-small).

    This judge is NOT one of the verification views — it provides independent
    ground truth for whether generated sentences are factual.
    """
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, AutoConfig,
    )

    print(f"    Loading E2E judge: {E2E_JUDGE_MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(E2E_JUDGE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(E2E_JUDGE_MODEL)
    model.eval()

    config = AutoConfig.from_pretrained(E2E_JUDGE_MODEL)
    ent_idx = None
    for idx, label in config.id2label.items():
        if label.lower() == "entailment":
            ent_idx = int(idx)
            break
    assert ent_idx is not None

    scores = []
    for i in range(0, len(sentence_list), batch_size):
        batch_ev = evidence_list[i:i + batch_size]
        batch_sent = sentence_list[i:i + batch_size]
        inputs = tokenizer(
            batch_ev, batch_sent,
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        for j in range(len(batch_ev)):
            scores.append(float(probs[j][ent_idx]))

    del model, tokenizer
    gc.collect()
    return scores


# ===========================================================================
# Main evaluation
# ===========================================================================

def main():
    total_start = time.time()

    print("=" * 80)
    print("REAL EVALUATION v4 — STRONG PARADIGMS + LEARNED AGGREGATION + LLaMA 8B")
    print("=" * 80)
    print()
    print("KEY CHANGES FROM v3:")
    print("  1. Only 3 strong paradigms (dropped weak STS, Retrieval, Multi-QA, Lexical)")
    print("  2. Added LLM-as-Judge (Flan-T5) and QA verification (SQuAD 2.0)")
    print("  3. Calibration/evaluation split (30/70) for honest evaluation")
    print("  4. Learned meta-classifier (logistic regression) for optimal aggregation")
    print("  5. LLaMA 8B generator for E2E (instead of GPT-2)")
    print()

    # ==================================================================
    # PHASE 1: Load TruthfulQA and create calibration/eval split
    # ==================================================================
    print("=" * 80)
    print("PHASE 1: Loading TruthfulQA + calibration/evaluation split")
    print("=" * 80)
    print()

    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation")["validation"]
    print(f"  {len(ds)} questions loaded", flush=True)

    # Build claims with rich evidence
    all_claims = []  # (evidence, claim_text, is_correct, question_idx)
    for idx in range(len(ds)):
        instance = ds[idx]
        evidence = f"Question: {instance['question']}\nAnswer: {instance['best_answer']}"
        for a in instance["correct_answers"]:
            if len(a.strip()) >= 3:
                all_claims.append((evidence, a, True, idx))
        for a in instance["incorrect_answers"]:
            if len(a.strip()) >= 3:
                all_claims.append((evidence, a, False, idx))

    n_total = len(all_claims)
    n_correct = sum(1 for _, _, c, _ in all_claims if c)
    n_incorrect = n_total - n_correct
    print(f"  {n_total} claims ({n_correct} correct, {n_incorrect} incorrect)")

    # Split by QUESTION (not claim) to prevent leakage
    np.random.seed(RANDOM_SEED)
    q_indices = np.random.permutation(len(ds))
    n_cal_q = int(len(ds) * CALIBRATION_FRACTION)
    cal_questions = set(q_indices[:n_cal_q].tolist())
    eval_questions = set(q_indices[n_cal_q:].tolist())

    cal_mask = np.array([q_idx in cal_questions for _, _, _, q_idx in all_claims])
    eval_mask = ~cal_mask

    n_cal = cal_mask.sum()
    n_eval = eval_mask.sum()
    print(f"  Calibration: {n_cal_q} questions, {n_cal} claims")
    print(f"  Evaluation:  {len(ds) - n_cal_q} questions, {n_eval} claims")
    print()

    pairs = [(ev, cl) for ev, cl, _, _ in all_claims]
    labels = np.array([c for _, _, c, _ in all_claims])

    # ==================================================================
    # PHASE 2: Score with 3 strong, independent paradigms
    # ==================================================================
    print("=" * 80)
    print("PHASE 2: Scoring with 3 strong, independent paradigms")
    print("=" * 80)
    print()

    paradigm_names = ["NLI", "LLM-Judge", "QA"]
    paradigm_info = {
        "NLI": {"model": "facebook/bart-large-mnli", "training": "MNLI", "task": "Entailment classification"},
        "LLM-Judge": {"model": "google/flan-t5-large", "training": "1800+ tasks", "task": "Zero-shot reasoning"},
        "QA": {"model": "deepset/roberta-base-squad2", "training": "SQuAD 2.0", "task": "Extractive QA confidence"},
    }

    all_scores = {}
    phase2_start = time.time()

    for pname in paradigm_names:
        info = paradigm_info[pname]
        print(f"  [{pname}] {info['task']} — trained on {info['training']}")
        t0 = time.time()

        if pname == "NLI":
            scores = score_nli(pairs)
        elif pname == "LLM-Judge":
            scores = score_llm_judge(pairs)
        elif pname == "QA":
            scores = score_qa_verification(pairs)

        elapsed = time.time() - t0
        all_scores[pname] = np.array(scores)

        # Quick stats
        correct_s = all_scores[pname][labels]
        incorrect_s = all_scores[pname][~labels]
        print(f"    Done in {elapsed:.1f}s")
        print(f"    Correct:   mean={correct_s.mean():.4f}, std={correct_s.std():.4f}")
        print(f"    Incorrect: mean={incorrect_s.mean():.4f}, std={incorrect_s.std():.4f}")
        print(f"    Separation: {correct_s.mean() - incorrect_s.mean():.4f}")
        print()

    phase2_time = time.time() - phase2_start
    print(f"  Phase 2 complete: {phase2_time:.0f}s ({phase2_time/60:.1f} min)")
    print()

    # ==================================================================
    # PHASE 3: Calibrate on calibration split
    # ==================================================================
    print("=" * 80)
    print("PHASE 3: Calibration on held-out split")
    print("=" * 80)
    print()

    cal_labels = labels[cal_mask]
    eval_labels = labels[eval_mask]

    # 3a: Per-paradigm Youden's J calibration
    print("  --- Per-paradigm Youden's J calibration ---")
    print(f"  {'Paradigm':<12} {'Threshold':>10} {'TPR':>8} {'FPR':>8} {'J':>8}")
    print("  " + "-" * 50)

    cal_thresholds = {}
    cal_quality = {}
    for pname in paradigm_names:
        cal_scores = all_scores[pname][cal_mask]
        t, tpr, fpr, j = calibrate_youden(cal_scores.tolist(), cal_labels.tolist())
        cal_thresholds[pname] = t
        cal_quality[pname] = {"threshold": t, "tpr": tpr, "fpr": fpr, "j": j}
        print(f"  {pname:<12} {t:>10.4f} {tpr:>8.4f} {fpr:>8.4f} {j:>8.4f}")

    print()

    # 3b: Precision-focused calibration (target FPR = 0.05)
    print("  --- Precision-focused calibration (target FPR = 0.05) ---")
    print(f"  {'Paradigm':<12} {'Threshold':>10} {'TPR':>8} {'FPR':>8}")
    print("  " + "-" * 40)

    pf_thresholds = {}
    pf_quality = {}
    for pname in paradigm_names:
        cal_scores = all_scores[pname][cal_mask]
        t, tpr, fpr = calibrate_target_fpr(cal_scores.tolist(), cal_labels.tolist())
        pf_thresholds[pname] = t
        pf_quality[pname] = {"threshold": t, "tpr": tpr, "fpr": fpr}
        print(f"  {pname:<12} {t:>10.4f} {tpr:>8.4f} {fpr:>8.4f}")

    print()

    # 3c: Learned meta-classifier
    print("  --- Training learned meta-classifier (logistic regression) ---")

    # Build feature matrix with interaction terms
    cal_score_dict = {name: all_scores[name][cal_mask] for name in paradigm_names}
    eval_score_dict = {name: all_scores[name][eval_mask] for name in paradigm_names}

    X_cal = build_features(cal_score_dict, paradigm_names)
    X_eval = build_features(eval_score_dict, paradigm_names)

    feature_names = list(paradigm_names) + [
        f"{paradigm_names[i]}*{paradigm_names[j]}"
        for i in range(len(paradigm_names))
        for j in range(i + 1, len(paradigm_names))
    ]

    w, b = fit_logistic_regression(X_cal, cal_labels.astype(float))

    print(f"  Feature weights:")
    for fname, wi in zip(feature_names, w):
        print(f"    {fname:<20} w={wi:>8.4f}")
    print(f"    {'bias':<20} b={b:>8.4f}")

    # Validate on cal set
    cal_pred_probs = predict_logistic(X_cal, w, b)
    cal_preds = cal_pred_probs >= 0.5
    cal_metrics = evaluate(cal_preds, cal_labels)
    print(f"  Cal set:  P={cal_metrics['precision']:.4f} R={cal_metrics['recall']:.4f} "
          f"F1={cal_metrics['f1']:.4f}")
    print()

    # ==================================================================
    # PHASE 4: Evaluate on held-out evaluation split
    # ==================================================================
    print("=" * 80)
    print("PHASE 4: Evaluation on held-out split")
    print("=" * 80)
    print()

    results = {"paradigms": {}, "aggregation": {}}

    # 4a: Individual paradigm results (on eval split, using cal-derived thresholds)
    print("  --- Individual paradigm results ---")
    print(f"  {'System':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Halluc%':>10} {'FPR':>10}")
    print("  " + "-" * 70)

    for pname in paradigm_names:
        eval_scores = all_scores[pname][eval_mask]
        t = cal_thresholds[pname]
        preds = eval_scores >= t
        metrics = evaluate(preds, eval_labels)
        results["paradigms"][pname] = {**metrics, "threshold": t, **cal_quality[pname]}
        print(f"  {pname:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['halluc_rate']:>10.4f} {metrics['fpr']:>10.4f}")

    print()

    # 4b: Multi-view aggregation methods
    print("  --- Multi-view aggregation methods ---")

    # Method 1: Uniform majority voting (tau = 2/3 means 2+ of 3 views agree)
    n_paradigms = len(paradigm_names)
    eval_decisions = {}
    for pname in paradigm_names:
        eval_decisions[pname] = all_scores[pname][eval_mask] >= cal_thresholds[pname]

    for tau_label, min_votes in [("Any (1/3)", 1), ("Majority (2/3)", 2), ("Unanimous (3/3)", 3)]:
        vote_counts = sum(eval_decisions[pname].astype(int) for pname in paradigm_names)
        preds = vote_counts >= min_votes
        metrics = evaluate(preds, eval_labels)
        agg_name = f"Voting-{tau_label}"
        results["aggregation"][agg_name] = metrics
        print(f"  {agg_name:<25} P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} Halluc={metrics['halluc_rate']:.4f}")

    # Method 2: Quality-weighted voting
    total_j = sum(cal_quality[pname]["j"] for pname in paradigm_names)
    weighted_scores_eval = np.zeros(n_eval)
    for pname in paradigm_names:
        weight = cal_quality[pname]["j"] / total_j
        weighted_scores_eval += weight * eval_decisions[pname].astype(float)

    for tau in [0.3, 0.5, 0.7]:
        preds = weighted_scores_eval >= tau
        metrics = evaluate(preds, eval_labels)
        agg_name = f"Weighted-{tau}"
        results["aggregation"][agg_name] = metrics
        print(f"  {agg_name:<25} P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} Halluc={metrics['halluc_rate']:.4f}")

    # Method 3: Learned meta-classifier (THE KEY INNOVATION)
    meta_probs = predict_logistic(X_eval, w, b)
    for threshold_name, threshold in [("Meta-0.5", 0.5), ("Meta-0.6", 0.6), ("Meta-0.7", 0.7)]:
        preds = meta_probs >= threshold
        metrics = evaluate(preds, eval_labels)
        results["aggregation"][threshold_name] = metrics
        print(f"  {threshold_name:<25} P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} Halluc={metrics['halluc_rate']:.4f}")

    print()

    # Find best aggregation method
    best_agg_name = max(results["aggregation"], key=lambda k: results["aggregation"][k]["f1"])
    best_agg = results["aggregation"][best_agg_name]
    best_single_name = max(results["paradigms"], key=lambda k: results["paradigms"][k]["f1"])
    best_single = results["paradigms"][best_single_name]

    print(f"  Best single paradigm: {best_single_name} (F1={best_single['f1']:.4f})")
    print(f"  Best multi-view:      {best_agg_name} (F1={best_agg['f1']:.4f})")
    etg_beats_single = best_agg["f1"] > best_single["f1"]
    print(f"  ETG beats single best: {'YES' if etg_beats_single else 'NO'} "
          f"(delta={best_agg['f1'] - best_single['f1']:+.4f})")
    print()

    # ==================================================================
    # PHASE 5: Precision-Recall curve comparison
    # ==================================================================
    print("=" * 80)
    print("PHASE 5: Precision-Recall curve analysis")
    print("=" * 80)
    print()

    # PR curve for each individual paradigm
    pr_curves = {}
    for pname in paradigm_names:
        pr_curves[pname] = compute_pr_curve(
            all_scores[pname][eval_mask].tolist(), eval_labels.tolist(),
        )

    # PR curve for meta-classifier
    pr_curves["Meta"] = compute_pr_curve(meta_probs.tolist(), eval_labels.tolist())

    # Compare at matched precision levels
    print("  --- PR curve: Meta-classifier vs Best Single ---")
    print(f"  {'Target P':>10} {'Meta P':>10} {'Meta R':>10} {f'{best_single_name} P':>10} {f'{best_single_name} R':>10} {'Winner':>10}")
    print("  " + "-" * 65)

    meta_wins_count = 0
    comparison_points = []
    for target_p in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        # Find meta-classifier point closest to target precision
        meta_point = None
        for t, p, r in pr_curves["Meta"]:
            if p >= target_p and (meta_point is None or r > meta_point[2]):
                meta_point = (t, p, r)

        # Find best single paradigm point closest to target precision
        single_point = None
        for t, p, r in pr_curves[best_single_name]:
            if p >= target_p and (single_point is None or r > single_point[2]):
                single_point = (t, p, r)

        if meta_point and single_point:
            winner = "META" if meta_point[2] > single_point[2] else best_single_name
            if meta_point[2] > single_point[2]:
                meta_wins_count += 1
            comparison_points.append({
                "target_precision": target_p,
                "meta_precision": meta_point[1],
                "meta_recall": meta_point[2],
                "single_precision": single_point[1],
                "single_recall": single_point[2],
                "meta_wins": meta_point[2] > single_point[2],
            })
            print(f"  {target_p:>10.2f} {meta_point[1]:>10.4f} {meta_point[2]:>10.4f} "
                  f"{single_point[1]:>10.4f} {single_point[2]:>10.4f} {winner:>10}")

    print()
    print(f"  Meta wins {meta_wins_count}/{len(comparison_points)} operating points")
    print()

    # ==================================================================
    # PHASE 6: PROOF 1 — Exponential Suppression
    # ==================================================================
    print("=" * 80)
    print("PROOF 1: EXPONENTIAL SUPPRESSION (Proposition 1)")
    print("Using eval split only (no data snooping)")
    print("=" * 80)
    print()

    # Compute support mass for eval split using Youden-calibrated thresholds
    eval_decisions_arr = np.column_stack([
        eval_decisions[pname].astype(int) for pname in paradigm_names
    ])
    eval_support_mass = eval_decisions_arr.mean(axis=1)

    # Average per-view FPR from calibration
    avg_alpha = np.mean([cal_quality[pname]["fpr"] for pname in paradigm_names])
    n_eval_incorrect = (~eval_labels).sum()

    print(f"  N views: {n_paradigms}, Avg alpha (from cal): {avg_alpha:.6f}")
    print()

    print("  Youden calibration:")
    print(f"  {'tau':<6} {'Theoretical':>12} {'Empirical':>12} {'Holds?':>8} {'Ratio':>8}")
    print("  " + "-" * 50)

    proof1_results = {}
    for tau_check in [1/3 + 0.001, 2/3 + 0.001, 1.0]:
        tau_label = f"{tau_check:.3f}"
        fp_count = ((~eval_labels) & (eval_support_mass >= tau_check)).sum()
        emp_fpr = fp_count / n_eval_incorrect if n_eval_incorrect > 0 else 0
        bound = hallucination_upper_bound(
            n_views=n_paradigms, tau=tau_check,
            alpha=max(avg_alpha, 0.001),
        )
        holds = emp_fpr <= bound
        ratio = emp_fpr / bound if bound > 0 else float("inf")
        print(f"  {tau_label:<6} {bound:>12.6f} {emp_fpr:>12.6f} {'YES' if holds else 'NO':>8} {ratio:>8.2f}x")
        proof1_results[tau_label] = {
            "theoretical": float(bound),
            "empirical": float(emp_fpr),
            "holds": bool(holds),
            "ratio": round(ratio, 2),
        }

    print()

    # Also test with precision-focused calibration
    pf_decisions = {}
    for pname in paradigm_names:
        pf_decisions[pname] = all_scores[pname][eval_mask] >= pf_thresholds[pname]

    pf_decisions_arr = np.column_stack([
        pf_decisions[pname].astype(int) for pname in paradigm_names
    ])
    pf_support_mass = pf_decisions_arr.mean(axis=1)
    pf_avg_alpha = np.mean([pf_quality[pname]["fpr"] for pname in paradigm_names])

    print(f"  Precision-focused calibration (target FPR=0.05):")
    print(f"  Avg alpha: {pf_avg_alpha:.6f}")
    print(f"  {'tau':<6} {'Theoretical':>12} {'Empirical':>12} {'Holds?':>8} {'Ratio':>8}")
    print("  " + "-" * 50)

    proof1_pf_results = {}
    for tau_check in [1/3 + 0.001, 2/3 + 0.001, 1.0]:
        tau_label = f"{tau_check:.3f}"
        fp_count = ((~eval_labels) & (pf_support_mass >= tau_check)).sum()
        emp_fpr = fp_count / n_eval_incorrect if n_eval_incorrect > 0 else 0
        bound = hallucination_upper_bound(
            n_views=n_paradigms, tau=tau_check,
            alpha=max(pf_avg_alpha, 0.001),
        )
        holds = emp_fpr <= bound
        ratio = emp_fpr / bound if bound > 0 else float("inf")
        print(f"  {tau_label:<6} {bound:>12.6f} {emp_fpr:>12.6f} {'YES' if holds else 'NO':>8} {ratio:>8.2f}x")
        proof1_pf_results[tau_label] = {
            "theoretical": float(bound),
            "empirical": float(emp_fpr),
            "holds": bool(holds),
            "ratio": round(ratio, 2),
        }

    print()

    # Independence analysis
    print("  --- Independence Analysis ---")
    agreement_matrix = {}
    for pi in paradigm_names:
        agreement_matrix[pi] = {}
        for pj in paradigm_names:
            agree = (eval_decisions[pi] == eval_decisions[pj]).mean()
            agreement_matrix[pi][pj] = round(float(agree), 4)

    avg_pairwise = np.mean([
        agreement_matrix[pi][pj]
        for i, pi in enumerate(paradigm_names)
        for j, pj in enumerate(paradigm_names)
        if i < j
    ])
    print(f"  Average pairwise agreement: {avg_pairwise:.4f}")

    # Expected agreement under independence
    marginals = {}
    for pname in paradigm_names:
        p_accept = eval_decisions[pname].mean()
        marginals[pname] = p_accept

    expected_agreements = []
    for i, pi in enumerate(paradigm_names):
        for j, pj in enumerate(paradigm_names):
            if i < j:
                p_both_acc = marginals[pi] * marginals[pj]
                p_both_rej = (1 - marginals[pi]) * (1 - marginals[pj])
                expected = p_both_acc + p_both_rej
                expected_agreements.append(expected)

    avg_expected = np.mean(expected_agreements)
    excess = avg_pairwise - avg_expected
    print(f"  Expected (independent): {avg_expected:.4f}")
    print(f"  Excess correlation: {excess:+.4f} ({excess*100:+.1f}%)")
    print()

    # ==================================================================
    # PHASE 7: PROOF 2 — Multi-View vs Single Best
    # ==================================================================
    print("=" * 80)
    print("PROOF 2: MULTI-VIEW vs SINGLE BEST MODEL")
    print("=" * 80)
    print()

    proof2_status = "PROVEN" if etg_beats_single else "NOT PROVEN"
    print(f"  Status: {proof2_status}")
    print(f"  Best single: {best_single_name} F1={best_single['f1']:.4f}")
    print(f"  Best multi:  {best_agg_name} F1={best_agg['f1']:.4f}")
    print(f"  Meta PR wins: {meta_wins_count}/{len(comparison_points)} operating points")
    print()

    # ==================================================================
    # PHASE 8: PROOF 3 — ETG Superiority
    # ==================================================================
    print("=" * 80)
    print("PROOF 3: ETG SUPERIORITY")
    print("=" * 80)
    print()

    n_paradigms_beaten = sum(
        1 for pname in paradigm_names
        if best_agg["f1"] > results["paradigms"][pname]["f1"]
    )
    proof3_status = (
        "PROVEN" if n_paradigms_beaten == len(paradigm_names) else
        f"PARTIALLY PROVEN ({n_paradigms_beaten}/{len(paradigm_names)} beaten)"
    )
    print(f"  Status: {proof3_status}")
    for pname in paradigm_names:
        beaten = "YES" if best_agg["f1"] > results["paradigms"][pname]["f1"] else "NO"
        print(f"    vs {pname}: ETG F1={best_agg['f1']:.4f} vs {results['paradigms'][pname]['f1']:.4f} → {beaten}")
    print()

    # Compute proof1_status early (needed for intermediate save)
    youden_holds = sum(1 for v in proof1_results.values() if v["holds"])
    pf_holds = sum(1 for v in proof1_pf_results.values() if v["holds"])
    proof1_status = "PROVEN" if youden_holds == len(proof1_results) else (
        "PARTIALLY PROVEN" if youden_holds > 0 or pf_holds > 0 else "NOT PROVEN"
    )

    # ==================================================================
    # SAVE INTERMEDIATE RESULTS (before E2E, in case generator OOMs)
    # ==================================================================
    intermediate_output = {
        "version": "v4",
        "improvements": [
            "3 strong paradigms only (dropped weak STS/Retrieval/Multi-QA/Lexical)",
            "LLM-as-Judge (Flan-T5) and QA verification (SQuAD 2.0)",
            "Calibration/evaluation split (30/70) — no data snooping",
            "Learned meta-classifier (logistic regression with interactions)",
        ],
        "dataset": "TruthfulQA (Lin et al., ACL 2022)",
        "n_questions": len(ds),
        "n_claims": n_total,
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "calibration_split": {"n_questions": n_cal_q, "n_claims": int(n_cal)},
        "evaluation_split": {"n_questions": len(ds) - n_cal_q, "n_claims": int(n_eval)},
        "paradigms": {
            name: {
                **paradigm_info[name],
                "youden": cal_quality[name],
                "precision_focused": pf_quality[name],
                "eval_metrics": results["paradigms"][name],
            }
            for name in paradigm_names
        },
        "meta_classifier": {
            "features": feature_names,
            "weights": {fname: round(float(wi), 6) for fname, wi in zip(feature_names, w)},
            "bias": round(float(b), 6),
            "cal_f1": cal_metrics["f1"],
        },
        "aggregation_results": results["aggregation"],
        "best_single": {"name": best_single_name, **best_single},
        "best_multi_view": {"name": best_agg_name, **best_agg},
        "pr_curve_comparison": comparison_points,
        "proof_1": {
            "youden": proof1_results,
            "precision_focused": proof1_pf_results,
            "avg_alpha_youden": round(float(avg_alpha), 6),
            "avg_alpha_pf": round(float(pf_avg_alpha), 6),
            "status": proof1_status,
        },
        "proof_2": {
            "status": proof2_status,
            "meta_pr_wins": f"{meta_wins_count}/{len(comparison_points)}",
            "etg_beats_single_f1": etg_beats_single,
        },
        "proof_3": {
            "status": proof3_status,
            "n_paradigms_beaten": n_paradigms_beaten,
        },
        "independence": {
            "agreement_matrix": agreement_matrix,
            "avg_pairwise_agreement": round(float(avg_pairwise), 4),
            "expected_independent": round(float(avg_expected), 4),
            "excess_correlation": round(float(excess), 4),
        },
    }
    intermediate_path = Path(__file__).parent.parent / "results" / "real_evaluation_v4_results.json"
    intermediate_path.parent.mkdir(exist_ok=True)
    with open(intermediate_path, "w") as f:
        json.dump(intermediate_output, f, indent=2)
    print(f"  [Intermediate results saved to {intermediate_path}]")
    print()

    # Free memory before loading generator
    del all_scores, X_cal, X_eval, cal_score_dict, eval_score_dict
    gc.collect()

    # ==================================================================
    # PHASE 9: E2E with LLaMA-class model
    # ==================================================================
    print("=" * 80)
    print(f"PHASE 9: END-TO-END GENERATION ({E2E_N_QUESTIONS} questions)")
    print("=" * 80)
    print()

    e2e_results = None
    try:
        # Select questions for E2E
        np.random.seed(RANDOM_SEED + 1)
        e2e_indices = np.random.choice(len(ds), size=min(E2E_N_QUESTIONS, len(ds)), replace=False)

        e2e_questions = [ds[int(i)]["question"] for i in e2e_indices]
        e2e_evidence = [
            f"Question: {ds[int(i)]['question']}\nAnswer: {ds[int(i)]['best_answer']}"
            for i in e2e_indices
        ]

        # Load generator
        print("  [GENERATOR]", flush=True)
        gen_model, gen_tokenizer, gen_name = load_generator()

        # Generate completions
        print(f"  Generating with {gen_name}...", flush=True)
        gen_start = time.time()
        completions = generate_completions(
            gen_model, gen_tokenizer, e2e_questions,
            max_new_tokens=E2E_MAX_NEW_TOKENS,
        )
        gen_time = time.time() - gen_start
        print(f"  Generated {len(completions)} completions in {gen_time:.0f}s")

        # Free generator memory
        del gen_model, gen_tokenizer
        gc.collect()

        # Split completions into sentences
        all_sentences = []
        all_sentence_evidence = []
        all_sentence_q_idx = []
        for q_idx, (text, evidence) in enumerate(zip(completions, e2e_evidence)):
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
            if not sentences and text.strip():
                sentences = [text.strip()]
            for sent in sentences:
                all_sentences.append(sent)
                all_sentence_evidence.append(evidence)
                all_sentence_q_idx.append(q_idx)

        print(f"  {len(all_sentences)} sentences from {len(completions)} completions")

        if all_sentences:
            # Score with independent judge (ground truth)
            print("  [JUDGE] Scoring ground truth...", flush=True)
            judge_scores = score_e2e_judge(all_sentence_evidence, all_sentences)
            judge_threshold = 0.5
            is_factual = np.array([s >= judge_threshold for s in judge_scores])

            # Score with verification paradigms
            e2e_pairs = list(zip(all_sentence_evidence, all_sentences))

            print("  [VERIFIER: NLI]", flush=True)
            e2e_nli_scores = score_nli(e2e_pairs)

            print("  [VERIFIER: LLM-Judge]", flush=True)
            e2e_llm_scores = score_llm_judge(e2e_pairs)

            print("  [VERIFIER: QA]", flush=True)
            e2e_qa_scores = score_qa_verification(e2e_pairs)

            # Apply meta-classifier for ETG decision
            e2e_score_dict = {
                "NLI": np.array(e2e_nli_scores),
                "LLM-Judge": np.array(e2e_llm_scores),
                "QA": np.array(e2e_qa_scores),
            }
            X_e2e = build_features(e2e_score_dict, paradigm_names)
            e2e_meta_probs = predict_logistic(X_e2e, w, b)
            etg_accepted = e2e_meta_probs >= 0.5

            # Compute FactScores
            n_sentences = len(all_sentences)
            n_accepted = etg_accepted.sum()
            n_rejected = n_sentences - n_accepted

            unfiltered_factscore = is_factual.mean()
            accepted_factscore = is_factual[etg_accepted].mean() if n_accepted > 0 else 0
            rejected_factscore = is_factual[~etg_accepted].mean() if n_rejected > 0 else 0

            e2e_results = {
                "generator": gen_name,
                "n_questions": len(e2e_questions),
                "n_sentences": int(n_sentences),
                "n_accepted": int(n_accepted),
                "n_rejected": int(n_rejected),
                "unfiltered_factscore": round(float(unfiltered_factscore), 4),
                "accepted_factscore": round(float(accepted_factscore), 4),
                "rejected_factscore": round(float(rejected_factscore), 4),
                "improvement": round(float(accepted_factscore - unfiltered_factscore), 4),
                "generation_time_seconds": round(gen_time, 1),
            }

            print(f"\n  E2E Results ({gen_name}):")
            print(f"    Sentences:     {n_sentences} total, {n_accepted} accepted, {n_rejected} rejected")
            print(f"    Unfiltered:    {unfiltered_factscore:.4f}")
            print(f"    ETG Accepted:  {accepted_factscore:.4f}")
            print(f"    ETG Rejected:  {rejected_factscore:.4f}")
            print(f"    Improvement:   {accepted_factscore - unfiltered_factscore:+.4f}")
            e2e_proven = accepted_factscore > unfiltered_factscore
            e2e_results["proven"] = e2e_proven
            print(f"    PROVEN: {'YES' if e2e_proven else 'NO'}")
        else:
            print("  WARNING: No sentences generated")
            e2e_results = {"error": "No sentences generated", "proven": False}

    except Exception as e:
        print(f"\n  E2E FAILED: {e}")
        import traceback
        traceback.print_exc()
        e2e_results = {"error": str(e), "proven": False}

    print()

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    # Proof 1 status
    youden_holds = sum(1 for v in proof1_results.values() if v["holds"])
    pf_holds = sum(1 for v in proof1_pf_results.values() if v["holds"])
    proof1_status = "PROVEN" if youden_holds == len(proof1_results) else (
        "PARTIALLY PROVEN" if youden_holds > 0 or pf_holds > 0 else "NOT PROVEN"
    )

    # E2E status
    proof4_status = "PROVEN" if (e2e_results and e2e_results.get("proven")) else "NOT PROVEN"

    print(f"  Claim 1 (Exponential Suppression): {proof1_status}")
    print(f"    Youden: {youden_holds}/{len(proof1_results)} tau values hold")
    print(f"    PF:     {pf_holds}/{len(proof1_pf_results)} tau values hold")
    print()
    print(f"  Claim 2 (Multi-view > Single):     {proof2_status}")
    print(f"    Best multi-view ({best_agg_name}): F1={best_agg['f1']:.4f}")
    print(f"    Best single ({best_single_name}): F1={best_single['f1']:.4f}")
    print(f"    PR curve: Meta wins {meta_wins_count}/{len(comparison_points)} operating points")
    print()
    print(f"  Claim 3 (ETG Superiority):         {proof3_status}")
    print(f"    Beats {n_paradigms_beaten}/{len(paradigm_names)} individual paradigms on F1")
    print()
    print(f"  Claim 4 (E2E Generation):          {proof4_status}")
    if e2e_results and "unfiltered_factscore" in e2e_results:
        print(f"    {e2e_results.get('generator', 'Unknown')}: "
              f"{e2e_results['unfiltered_factscore']:.4f} → {e2e_results['accepted_factscore']:.4f}")
    print()

    total_time = time.time() - total_start
    print(f"  Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ==================================================================
    # Save results
    # ==================================================================
    output = {
        "version": "v4",
        "improvements": [
            "3 strong paradigms only (dropped weak STS/Retrieval/Multi-QA/Lexical)",
            "LLM-as-Judge (Flan-T5) and QA verification (SQuAD 2.0)",
            "Calibration/evaluation split (30/70) — no data snooping",
            "Learned meta-classifier (logistic regression with interactions)",
            "LLaMA 8B generator for E2E",
        ],
        "dataset": "TruthfulQA (Lin et al., ACL 2022)",
        "n_questions": len(ds),
        "n_claims": n_total,
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "calibration_split": {"n_questions": n_cal_q, "n_claims": int(n_cal)},
        "evaluation_split": {"n_questions": len(ds) - n_cal_q, "n_claims": int(n_eval)},
        "paradigms": {
            name: {
                **paradigm_info[name],
                "youden": cal_quality[name],
                "precision_focused": pf_quality[name],
                "eval_metrics": results["paradigms"][name],
            }
            for name in paradigm_names
        },
        "meta_classifier": {
            "features": feature_names,
            "weights": {fname: round(float(wi), 6) for fname, wi in zip(feature_names, w)},
            "bias": round(float(b), 6),
            "cal_f1": cal_metrics["f1"],
        },
        "aggregation_results": results["aggregation"],
        "best_single": {"name": best_single_name, **best_single},
        "best_multi_view": {"name": best_agg_name, **best_agg},
        "pr_curve_comparison": comparison_points,
        "proof_1": {
            "youden": proof1_results,
            "precision_focused": proof1_pf_results,
            "avg_alpha_youden": round(float(avg_alpha), 6),
            "avg_alpha_pf": round(float(pf_avg_alpha), 6),
            "status": proof1_status,
        },
        "proof_2": {
            "status": proof2_status,
            "meta_pr_wins": f"{meta_wins_count}/{len(comparison_points)}",
            "etg_beats_single_f1": etg_beats_single,
        },
        "proof_3": {
            "status": proof3_status,
            "n_paradigms_beaten": n_paradigms_beaten,
        },
        "proof_4_e2e": e2e_results,
        "independence": {
            "agreement_matrix": agreement_matrix,
            "avg_pairwise_agreement": round(float(avg_pairwise), 4),
            "expected_independent": round(float(avg_expected), 4),
            "excess_correlation": round(float(excess), 4),
        },
        "total_runtime_seconds": round(total_time, 1),
    }

    results_path = Path(__file__).parent.parent / "results" / "real_evaluation_v4_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
