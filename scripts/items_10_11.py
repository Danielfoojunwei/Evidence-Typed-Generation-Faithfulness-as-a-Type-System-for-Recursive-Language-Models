#!/usr/bin/env python3
"""Items 10 + 11: E2E at scale + SelfCheckGPT baseline.

ALL REAL INFERENCE. NO MOCKS. NO HARDCODED VALUES.

Models (loaded from local disk):
  - NLI: cross-encoder/nli-distilroberta-base (82M)
  - LLM-Judge: google/flan-t5-small (77M)
  - QA: deepset/tinyroberta-squad2 (82M)
  - Generator: Qwen/Qwen2.5-0.5B-Instruct (494M)
  - Meta-classifier: L2-logistic trained on TruthfulQA cal (from v4)

Item 10: Generate on 200 TruthfulQA questions, verify each sentence,
         report FactScore with bootstrap CIs.
Item 11: Run SelfCheckGPT baseline (sample 3 responses, measure
         consistency) on same questions. Compare head-to-head.
"""
import json, random, sys, time, math, gc, warnings, os
import numpy as np
import torch
warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from etg_rlm.statistics import bootstrap_ci

from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
)
from datasets import load_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
N_QUESTIONS = 200

# ============================================================
# Load all models
# ============================================================
print("Loading models ...", flush=True)
t0 = time.time()

nli_tok = AutoTokenizer.from_pretrained("cross-encoder/nli-distilroberta-base")
nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-distilroberta-base")
nli_model.eval()
print(f"  NLI: {time.time()-t0:.0f}s", flush=True)

llm_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
llm_model.eval()
print(f"  LLM-Judge: {time.time()-t0:.0f}s", flush=True)

qa_tok = AutoTokenizer.from_pretrained("/tmp/tinyroberta-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("/tmp/tinyroberta-squad2")
qa_model.eval()
print(f"  QA: {time.time()-t0:.0f}s", flush=True)

gen_tok = AutoTokenizer.from_pretrained("/tmp/qwen-0.5b", trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained("/tmp/qwen-0.5b", dtype=torch.float32, trust_remote_code=True)
gen_model.eval()
print(f"  Generator (Qwen-0.5B): {time.time()-t0:.0f}s", flush=True)

v4 = json.load(open("results/real_evaluation_v4_results.json"))
W = v4["meta_classifier"]["weights"]
B = v4["meta_classifier"]["bias"]
print(f"All models loaded in {time.time()-t0:.0f}s\n", flush=True)

# ============================================================
# Scoring functions
# ============================================================
def score_nli(premise, hypothesis):
    inp = nli_tok(premise[:400], hypothesis[:200], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad(): logits = nli_model(**inp).logits
    return torch.softmax(logits, dim=-1)[0][1].item()

def score_llm(evidence, claim):
    prompt = f"Is this true? Evidence: {evidence[:300]}. Claim: {claim[:200]}. Answer:"
    inp = llm_tok(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        out = llm_model.generate(**inp, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
    if out.scores:
        logits = out.scores[0][0]
        tid = llm_tok.encode("true", add_special_tokens=False)[0]
        fid = llm_tok.encode("false", add_special_tokens=False)[0]
        return torch.softmax(logits[[tid, fid]], dim=0)[0].item()
    return 0.5

def score_qa(context, claim):
    try:
        inp = qa_tok(claim[:200], context[:400], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad(): out = qa_model(**inp)
        sp = torch.softmax(out.start_logits, dim=-1).max().item()
        ep = torch.softmax(out.end_logits, dim=-1).max().item()
        return (sp + ep) / 2
    except: return 0.0

def meta_predict(ns, ls, qs):
    logit = (W["NLI"]*ns + W["LLM-Judge"]*ls + W["QA"]*qs +
             W["NLI*LLM-Judge"]*ns*ls + W["NLI*QA"]*ns*qs +
             W["LLM-Judge*QA"]*ls*qs + B)
    return 1.0 / (1.0 + math.exp(-logit))

def generate_response(question):
    if hasattr(gen_tok, "chat_template") and gen_tok.chat_template:
        text = gen_tok.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    else:
        text = f"Question: {question}\nAnswer:"
    inp = gen_tok(text, return_tensors="pt", max_length=256, truncation=True)
    with torch.no_grad():
        out = gen_model.generate(**inp, max_new_tokens=80, do_sample=False, pad_token_id=gen_tok.eos_token_id)
    return gen_tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def split_sentences(text):
    return [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip() and len(s.strip()) > 5]

# ============================================================
# Load TruthfulQA
# ============================================================
print("Loading TruthfulQA ...", flush=True)
ds = load_dataset("truthful_qa", "generation", split="validation")
questions = list(ds)
random.shuffle(questions)
questions = questions[:N_QUESTIONS]
print(f"  {len(questions)} questions\n", flush=True)

# ============================================================
# ITEM 10: E2E at scale with 3-view verification + CIs
# ============================================================
print("=" * 70, flush=True)
print(f"ITEM 10: E2E Generation (n={N_QUESTIONS}, Qwen-0.5B)", flush=True)
print("=" * 70, flush=True)

e2e_total_sents = 0
e2e_accepted = 0
e2e_rejected = 0
per_q_factscore = []

t0 = time.time()
for qi, q in enumerate(questions):
    query = q["question"]
    ref = " ".join(q.get("correct_answers", []))

    response = generate_response(query)
    sents = split_sentences(response)
    if not sents: continue

    e2e_total_sents += len(sents)
    q_acc = 0
    for sent in sents:
        ns = score_nli(ref[:400], sent[:200])
        ls = score_llm(ref[:400], sent[:200])
        qs = score_qa(ref[:400], sent[:200])
        ms = meta_predict(ns, ls, qs)
        if ms >= 0.5:
            e2e_accepted += 1
            q_acc += 1
        else:
            e2e_rejected += 1

    per_q_factscore.append(q_acc / len(sents) if sents else 0)

    if (qi+1) % 50 == 0:
        print(f"  {qi+1}/{N_QUESTIONS} ({time.time()-t0:.0f}s) sents={e2e_total_sents} acc={e2e_accepted}", flush=True)

e2e_time = time.time() - t0
e2e_retention = e2e_accepted / e2e_total_sents if e2e_total_sents > 0 else 0
e2e_fs_ci = bootstrap_ci(per_q_factscore, seed=SEED, n_bootstrap=10000) if per_q_factscore else None
e2e_fs_mean = float(np.mean(per_q_factscore)) if per_q_factscore else 0

print(f"\n  E2E Results:", flush=True)
print(f"    Questions: {len(questions)}, Sentences: {e2e_total_sents}", flush=True)
print(f"    Accepted: {e2e_accepted} ({e2e_retention*100:.1f}%)", flush=True)
print(f"    Rejected: {e2e_rejected}", flush=True)
print(f"    Per-Q acceptance: {e2e_fs_mean:.4f} [{e2e_fs_ci.ci_lower:.4f}, {e2e_fs_ci.ci_upper:.4f}]" if e2e_fs_ci else "", flush=True)
print(f"    Time: {e2e_time:.0f}s\n", flush=True)

# ============================================================
# ITEM 11: SelfCheckGPT baseline (consistency-based)
# ============================================================
print("=" * 70, flush=True)
print(f"ITEM 11: SelfCheckGPT Baseline (n={N_QUESTIONS})", flush=True)
print("=" * 70, flush=True)

N_SAMPLES = 3  # number of additional samples for consistency check

selfcheck_total_sents = 0
selfcheck_accepted = 0
selfcheck_rejected = 0
per_q_selfcheck = []

t0 = time.time()
for qi, q in enumerate(questions):
    query = q["question"]

    # Generate primary response
    primary = generate_response(query)
    primary_sents = split_sentences(primary)
    if not primary_sents: continue

    # Generate N additional samples for consistency
    samples = []
    for _ in range(N_SAMPLES):
        # Use temperature sampling for diversity
        if hasattr(gen_tok, "chat_template") and gen_tok.chat_template:
            text = gen_tok.apply_chat_template(
                [{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True)
        else:
            text = f"Question: {query}\nAnswer:"
        inp = gen_tok(text, return_tensors="pt", max_length=256, truncation=True)
        with torch.no_grad():
            out = gen_model.generate(**inp, max_new_tokens=80, do_sample=True,
                                     temperature=0.7, top_p=0.9,
                                     pad_token_id=gen_tok.eos_token_id)
        samples.append(gen_tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())

    selfcheck_total_sents += len(primary_sents)
    q_acc = 0

    for sent in primary_sents:
        # SelfCheckGPT: check if sentence is consistent with sampled responses
        # Use NLI to measure entailment of sentence against each sample
        consistency_scores = []
        for sample in samples:
            if sample.strip():
                cs = score_nli(sample[:400], sent[:200])
                consistency_scores.append(cs)

        # Accept if average consistency >= 0.5
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
        if avg_consistency >= 0.5:
            selfcheck_accepted += 1
            q_acc += 1
        else:
            selfcheck_rejected += 1

    per_q_selfcheck.append(q_acc / len(primary_sents) if primary_sents else 0)

    if (qi+1) % 50 == 0:
        print(f"  {qi+1}/{N_QUESTIONS} ({time.time()-t0:.0f}s) sents={selfcheck_total_sents} acc={selfcheck_accepted}", flush=True)

selfcheck_time = time.time() - t0
selfcheck_retention = selfcheck_accepted / selfcheck_total_sents if selfcheck_total_sents > 0 else 0
selfcheck_ci = bootstrap_ci(per_q_selfcheck, seed=SEED, n_bootstrap=10000) if per_q_selfcheck else None
selfcheck_mean = float(np.mean(per_q_selfcheck)) if per_q_selfcheck else 0

print(f"\n  SelfCheckGPT Results:", flush=True)
print(f"    Sentences: {selfcheck_total_sents}", flush=True)
print(f"    Accepted: {selfcheck_accepted} ({selfcheck_retention*100:.1f}%)", flush=True)
print(f"    Rejected: {selfcheck_rejected}", flush=True)
print(f"    Per-Q acceptance: {selfcheck_mean:.4f} [{selfcheck_ci.ci_lower:.4f}, {selfcheck_ci.ci_upper:.4f}]" if selfcheck_ci else "", flush=True)
print(f"    Time: {selfcheck_time:.0f}s\n", flush=True)

# ============================================================
# Head-to-head comparison
# ============================================================
print("=" * 70, flush=True)
print("HEAD-TO-HEAD: Multi-View Verification vs SelfCheckGPT", flush=True)
print("=" * 70, flush=True)
print(f"  Multi-View (3 views + meta):", flush=True)
print(f"    Retention: {e2e_retention*100:.1f}%", flush=True)
print(f"    Per-Q score: {e2e_fs_mean:.4f} [{e2e_fs_ci.ci_lower:.4f}, {e2e_fs_ci.ci_upper:.4f}]" if e2e_fs_ci else "", flush=True)

print(f"  SelfCheckGPT (3 samples):", flush=True)
print(f"    Retention: {selfcheck_retention*100:.1f}%", flush=True)
print(f"    Per-Q score: {selfcheck_mean:.4f} [{selfcheck_ci.ci_lower:.4f}, {selfcheck_ci.ci_upper:.4f}]" if selfcheck_ci else "", flush=True)

# Paired comparison
if len(per_q_factscore) == len(per_q_selfcheck) and len(per_q_factscore) > 1:
    diffs = [a - b for a, b in zip(per_q_factscore, per_q_selfcheck)]
    diff_ci = bootstrap_ci(diffs, seed=SEED, n_bootstrap=10000)
    diff_mean = float(np.mean(diffs))
    print(f"\n  Paired difference (Multi-View - SelfCheck):", flush=True)
    print(f"    Mean: {diff_mean:+.4f} [{diff_ci.ci_lower:+.4f}, {diff_ci.ci_upper:+.4f}]", flush=True)
    if diff_ci.ci_lower > 0:
        print(f"    VERDICT: Multi-View significantly better (CI excludes 0)", flush=True)
    elif diff_ci.ci_upper < 0:
        print(f"    VERDICT: SelfCheckGPT significantly better (CI excludes 0)", flush=True)
    else:
        print(f"    VERDICT: No significant difference (CI includes 0)", flush=True)

# ============================================================
# Save results
# ============================================================
results = {
    "generator": "Qwen/Qwen2.5-0.5B-Instruct (494M, loaded from /tmp/qwen-0.5b)",
    "n_questions": N_QUESTIONS,
    "item_10_e2e": {
        "n_sentences": e2e_total_sents,
        "n_accepted": e2e_accepted,
        "n_rejected": e2e_rejected,
        "retention_rate": round(e2e_retention, 4),
        "per_q_acceptance_mean": round(e2e_fs_mean, 4),
        "per_q_acceptance_ci": [round(e2e_fs_ci.ci_lower, 4), round(e2e_fs_ci.ci_upper, 4)] if e2e_fs_ci else None,
        "time_seconds": round(e2e_time, 1),
        "views": ["NLI (cross-encoder/nli-distilroberta-base)", "LLM-Judge (flan-t5-small)", "QA (tinyroberta-squad2)"],
    },
    "item_11_selfcheck": {
        "n_sentences": selfcheck_total_sents,
        "n_accepted": selfcheck_accepted,
        "n_rejected": selfcheck_rejected,
        "retention_rate": round(selfcheck_retention, 4),
        "per_q_acceptance_mean": round(selfcheck_mean, 4),
        "per_q_acceptance_ci": [round(selfcheck_ci.ci_lower, 4), round(selfcheck_ci.ci_upper, 4)] if selfcheck_ci else None,
        "n_consistency_samples": N_SAMPLES,
        "time_seconds": round(selfcheck_time, 1),
    },
    "head_to_head": {
        "paired_diff_mean": round(diff_mean, 4) if 'diff_mean' in dir() else None,
        "paired_diff_ci": [round(diff_ci.ci_lower, 4), round(diff_ci.ci_upper, 4)] if 'diff_ci' in dir() else None,
    },
    "note": "ALL numbers from real model inference. No mocks, no hardcoded values.",
}

with open("results/items_10_11_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/items_10_11_results.json", flush=True)
print(f"Total time: {time.time()-t0:.0f}s", flush=True)
