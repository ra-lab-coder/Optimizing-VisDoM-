
from __future__ import annotations
import argparse, json, os, math, re, itertools, textwrap, random, time
from collections import Counter, defaultdict
from typing import List, Dict, Any, Iterable, Tuple, Optional


def to_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def to_int(x, default=0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(float(x))
        except Exception:
            return default

def to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return str(x)
    if isinstance(x, (list, tuple)):
        return " ".join(to_text(y) for y in x)
    return str(x).strip()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk

def normalize_eval(input_path: str, correct_threshold: float = 0.5) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for i, (q, r) in enumerate(data.items(), start=1):
        f1 = to_float(r.get("Word-Overlap f1"))
        pred = to_text(r.get("answer"))
        gold = to_text(r.get("gt_answer"))
        src = to_text(r.get("source_filename"))

        rows.append({
            "qid": f"q{i:04d}",
            "question": to_text(q),
            "pred": pred,
            "gold": gold,
            "source": src,
            "f1": f1,
            "pred_len": to_int(r.get("answer_token_len"), 0),
            "gold_len": to_int(r.get("gt_answer_token_len"), 0),
            "is_missing_pred": (len(pred) == 0),
            "is_missing_f1": (f1 is None),
            "is_correct": (f1 is not None and f1 >= correct_threshold),
        })
    return rows


CORRECT_THRESHOLD = 0.5
GENERIC_MIN_TOKENS = 21
GENERIC_F1_MAX = 0.15
OVERLONG_MIN_MULT = 2.0
OVERLONG_MIN_ABS = 10
SEM_MIN_RECALL = 0.60
SEM_MAX_PRECISION = 0.50

INC = {"increase","increases","increased","increasing",
       "improve","improves","improved","improving",
       "rise","rises","rising","grow","grows","growth","higher","more","up"}
DEC = {"decrease","decreases","decreased","decreasing",
       "decline","declines","declined","declining",
       "drop","drops","dropped","dropping","fall","falls","fallen",
       "worse","worsen","worsens","lower","less","down"}

TOK_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def tok(s: str) -> list[str]:
    return TOK_RE.findall((s or "").lower())

def first_num(s: str):
    m = NUM_RE.search(s or "")
    return float(m.group()) if m else None

def contains_any(s: str, vocab: set[str]) -> bool:
    s = (s or "").lower()
    return any(w in s for w in vocab)


def is_direction_flip(gold: str, pred: str) -> bool:
    g_inc, g_dec = contains_any(gold, INC), contains_any(gold, DEC)
    p_inc, p_dec = contains_any(pred, INC), contains_any(pred, DEC)
    return (g_inc and p_dec) or (g_dec and p_inc)

def is_generic_offtopic(gold: str, pred: str, f1: float | None) -> bool:
    if f1 is None:
        return False
    return (len(tok(pred)) >= GENERIC_MIN_TOKENS) and (f1 < GENERIC_F1_MAX)

def is_correct_but_verbose_or_paraphrase(gold: str, pred: str, f1: float | None) -> bool:
    
    if f1 is None or f1 >= CORRECT_THRESHOLD:
        return False

    g_tokens, p_tokens = tok(gold), tok(pred)
    if not g_tokens or not p_tokens:
        return False

    g_set, p_set = set(g_tokens), set(p_tokens)
    inter = len(g_set & p_set)
    recall = inter / max(1, len(g_set))
    precision = inter / max(1, len(p_set))

    coverage = (" ".join(g_tokens) in " ".join(p_tokens)) or (recall >= SEM_MIN_RECALL)
    paraphrase_like = (recall >= SEM_MIN_RECALL) and (precision <= SEM_MAX_PRECISION)
    length_inflated = len(p_tokens) >= max(OVERLONG_MIN_ABS, int(OVERLONG_MIN_MULT * len(g_tokens)))
    verbose_correct = coverage and length_inflated
    verbose_correct_1 = paraphrase_like and length_inflated

    return verbose_correct or paraphrase_like or verbose_correct_1



def heuristic_bucket(item: Dict[str, Any]) -> str:
    """
    Returns one of:
      correct_but_vercose_or_paraphrase, empty_pred, direction_flip,
      numeric_off_by_many,
      generic_or_offtopic,
      entity_or_attribute_mismatch, other
    """
    gold = (item.get("gold") or "").strip()
    pred = (item.get("pred") or "").strip()
    f1 = item.get("f1")

    if (f1 is not None) and (f1 >= CORRECT_THRESHOLD):
        return "other"

    if pred == "":
        return "empty_pred"

    gnum, pnum = first_num(gold), first_num(pred)
    if gnum is not None and pnum is not None:
        diff = abs(pnum - gnum)
        if diff >= 2:
            return "numeric_off_by_many"

    if is_direction_flip(gold, pred):
        return "direction_flip"


    if is_generic_offtopic(gold, pred, f1):
        return "generic_or_offtopic"
    
    if is_correct_but_verbose_or_paraphrase(gold, pred, f1):
        return "correct_but_verbose_or_paraphrase"


    g_set, p_set = set(tok(gold)), set(tok(pred))
    inter = len(g_set & p_set)
    recall = inter / max(1, len(g_set))
    if recall < 0.4:
        return "entity_or_attribute_mismatch"

    return "other"


ERROR_BUCKET = [
    "correct_but_verbose_or_paraphrase — prediction is content-correct but F1 is low because it is either overly long (precision penalty) or uses different wording (synonyms/quantified restatement) that reduces token overlap.",
    "empty_pred — prediction is empty/whitespace only.",
    "insufficient_evidence - prediction names a specific answer, but the provided evidence is missing, irrelevant, or too weak to substantiate it (no direct support in the source)."
    "direction_flip — says increase vs decrease (or vice versa) for trends.",
    "numeric_off_by_many — BOTH gold and prediction contain numbers in the same unit; |pred−gold|≥2.",
    "generic_or_offtopic — long/hand-wavy text with little specific content.",
    "entity_or_attribute_mismatch — talks about the wrong entity/attribute/column.",
    "other — none of the above fits confidently."
]

ERROR_CAUSE = [
    "misread_trend","numeric_error","verbosity","contradiction",
    "lexical_metric_limit","ambiguous_gold","off_topic","unknown","n/a"
]

def build_llm_prompt(items: List[Dict[str, Any]]) -> Tuple[str, str]:
    sys = (
        "You label VQA-style QA pairs. Output STRICT JSON only (one object per line), no prose. "
        "Do NOT reveal chain-of-thought. Keep 'explanation_short' ≤ 15 words. "
        "If judging requires the image/figure, set flags.needs_image=true. "
        "Priority: empty_pred -> insufficient_evidence -> correct_but_verbose_or_paraphrase -> direction_flip -> numeric_off_by_many -> generic_or_offtopic -> entity_or_attribute_mismatch -> other. Use “other” ONLY if none of the above applies. If an item fits overlong_correct, do not choose generic_or_offtopic or other."
        f"bucket ∈ {ERROR_BUCKET}. root_cause ∈ {ERROR_CAUSE}."
    )
    header = "Label the following items:\n"
    body = "\n".join(json.dumps({
        "qid": it["qid"],
        "question": it["question"],
        "gold": it["gold"],
        "pred": it["pred"],
        "f1": (it["f1"] if it["f1"] is not None else None)
    }, ensure_ascii=False) for it in items)
    ask = (
        f"{header}{body}\n\n"
        "Return one JSON object per line with fields: "
        '{"qid":...,"bucket":...,"root_cause":...,"explanation_short":...,"flags":{"needs_image":true|false,"needs_review":true|false}}'
    )
    return sys, ask

def parse_llm_json_lines(text: str) -> List[Dict[str, Any]]:
    out = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("```"):
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "qid" in obj:
                out.append(obj)
        except json.JSONDecodeError:
            # try to salvage trailing commas or accidental comments
            s2 = s.rstrip(",")
            try:
                obj = json.loads(s2)
                if isinstance(obj, dict) and "qid" in obj:
                    out.append(obj)
            except json.JSONDecodeError:
                continue
    return out

def call_openai(system_prompt: str, user_prompt: str, model: str, openai_api_keys: str) -> str:
    """
    Minimal OpenAI Chat Completions call.
    Requires: `pip install openai` and OPENAI_API_KEY env var.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_keys)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")

def aggregate(labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_counts = Counter(l.get("bucket", "other") for l in labels)
    total = sum(bucket_counts.values())
    dist = [{"bucket": b, "n": n, "pct": round(100*n/total, 1)} for b, n in bucket_counts.most_common()]
    return {"total_labeled": total, "bucket_dist": dist}

def write_csv_distribution(path: str, dist: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as w:
        w.write("bucket,n,pct\n")
        for row in dist:
            w.write(f'{row["bucket"]},{row["n"]},{row["pct"]}\n')

def make_examples_by_bucket(labels: List[Dict[str, Any]], rows_by_qid: Dict[str, Dict[str, Any]], k:int=3):
    per_bucket = defaultdict(list)
    for lab in labels:
        per_bucket[lab["bucket"]].append(lab["qid"])
    out = {}
    for b, qids in per_bucket.items():
        picks = qids[:k]
        ex = []
        for qid in picks:
            r = rows_by_qid[qid]
            ex.append({
                "qid": qid,
                "question": r["question"],
                "gold": r["gold"],
                "pred": r["pred"],
                "f1": r["f1"],
                "explanation_short": next((l["explanation_short"] for l in labels if l["qid"]==qid), "")
            })
        out[b] = ex
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to your JSON (dict of Q → fields).")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--correct-threshold", type=float, default=0.5)
    ap.add_argument("--label-mode", choices=["none","heuristics","openai"], default="heuristics",
                    help="'none' just prepares JSONL; 'heuristics' runs offline labeling; 'openai' calls an LLM.")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model name for --label-mode openai.")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--only-incorrect", action="store_true", help="Label only incorrect/missing cases.")
    ap.add_argument("--max-items", type=int, default=0, help="Optional cap on number of items to label (0 = all).")
    ap.add_argument("--openai_api_keys", type=str)
    args = ap.parse_args()

    ensure_dir(args.out)

    # 1) Normalize
    rows = normalize_eval(args.input, args.correct_threshold)
    rows_by_qid = {r["qid"]: r for r in rows}

    total_questions = len(rows)
    incorrect_or_missing = sum(1 for r in rows if (r["is_missing_pred"] or r["is_missing_f1"] or not r["is_correct"]))
    correct = sum(1 for r in rows if r["is_correct"])

    # 2) Write JSONL (full + incorrect-only)
    full_path = os.path.join(args.out, "qa_eval_full.jsonl")
    inc_path  = os.path.join(args.out, "qa_eval_incorrect_only.jsonl")
    write_jsonl(full_path, rows)
    only_incorrect = [r for r in rows if (r["is_missing_pred"] or r["is_missing_f1"] or not r["is_correct"])]
    write_jsonl(inc_path, only_incorrect)

    print(f"Wrote {full_path} (N={len(rows)})")
    print(f"Wrote {inc_path} (N={len(only_incorrect)})")

    # 3) Labeling
    labels: List[Dict[str, Any]] = []
    items_to_label = only_incorrect if (args.only_incorrect or args.label_mode != "none") else rows
    if args.max_items and args.max_items > 0:
        items_to_label = items_to_label[:args.max_items]

    if args.label_mode == "none":
        summary = {"total_labeled": 0, "bucket_dist": []}
    elif args.label_mode == "openai":
        openai_api_keys = args.openai_api_keys
        for chunk in batched(items_to_label, args.batch_size):
            sys_prompt, user_prompt = build_llm_prompt(chunk)
            raw = call_openai(sys_prompt, user_prompt, model=args.model, openai_api_keys=openai_api_keys)
            parsed = parse_llm_json_lines(raw)
            # If model returns fewer lines than asked (it happens), fill rest with heuristics
            got_qids = {p["qid"] for p in parsed}
            for p in parsed:
                labels.append(p)
            for it in chunk:
                if it["qid"] not in got_qids:
                    labels.append(heuristic_bucket(it, args.correct_threshold))
    else:
        raise ValueError("Unknown label mode")
    
    for lab in labels:
        qid = lab["qid"]
        item = rows_by_qid[qid]
        if lab["bucket"] == "other" or lab["bucket"]== "generic_or_offtopic":
            bucket = heuristic_bucket(item)
            lab.update({"bucket":bucket})                
    
    # 4) Save labels and aggregate
    if labels:
        labels_path = os.path.join(args.out, "labels.jsonl")
        write_jsonl(labels_path, labels)
        print(f"[✓] Wrote labels → {labels_path} (N={len(labels)})")

    summary = aggregate(labels) if labels else {"total_labeled": 0, "bucket_dist": []}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as w:
        json.dump({
            "total_questions": total_questions,
            "incorrect_or_missing": incorrect_or_missing,
            "correct": correct,
            **summary
        }, w, ensure_ascii=False, indent=2)

    write_csv_distribution(os.path.join(args.out, "dist.csv"), summary.get("bucket_dist", []))

    examples = make_examples_by_bucket(labels, rows_by_qid, k=3) if labels else {}
    with open(os.path.join(args.out, "examples_by_bucket.json"), "w", encoding="utf-8") as w:
        json.dump(examples, w, ensure_ascii=False, indent=2)

    print("\n[Summary]")
    print(f"  total_questions       : {total_questions}")
    print(f"  correct (F1 ≥ {args.correct_threshold}) : {correct}")
    print(f"  incorrect_or_missing  : {incorrect_or_missing}")
    if labels:
        print(f"  labeled               : {summary['total_labeled']}")
        print(f"  top buckets           : {[ (d['bucket'], d['n']) for d in summary['bucket_dist'][:5] ]}")

if __name__ == "__main__":
    main()


