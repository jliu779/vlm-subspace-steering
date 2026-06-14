#!/usr/bin/env python3
"""Score MathVista (testmini) generations.

MathVista mixes multiple_choice (letter A–H) and free_form (integer/float/list)
answers, so score_scienceqa.py (A–E letter only) is insufficient. This scorer:
  - MCQ rows  (metadata.question_type=='multiple_choice'): letter match A–H vs
    metadata answer_letter.
  - free_form rows: heuristic value extraction from the response, normalised-match
    against metadata.answer keyed on metadata.answer_type (integer/float/list).
Refusals / un-parseable count as WRONG (honest 口径 = correct / N_total), matching
score_scienceqa.py. Free-form matching is a HEURISTIC (no GPT extractor); the MCQ
subset is exact. The CSV reports overall + per-subset accuracy so the reliable MCQ
number is separable from the heuristic free-form one.

Run:
  python scripts/judge/score_mathvista.py --generations <out.jsonl> \
      --manifest data/manifests/mathvista.jsonl --out <score.csv> --per_sample_out <judged.jsonl>
"""
import argparse, csv, json, re

LETTER = re.compile(r"answer\s*(?:is|:)?\s*\**\s*\(?([A-H])\b", re.I)
LETTER2 = re.compile(r"^\s*\(?([A-H])[\).\s]")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def first_letter(s):
    if not s: return None
    m = LETTER.search(s) or LETTER2.match(s)
    if m: return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", s)
    return m.group(1).upper() if m else None

def _norm_num(x):
    x = str(x).strip().replace(",", "").replace("$", "")
    x = re.sub(r"[a-zA-Z%°]+$", "", x).strip()
    try: return float(x)
    except ValueError: return None

def _nums(s):
    return [n.replace(",", "") for n in NUM.findall(s or "")]

def match_free(resp, gold, atype, precision):
    if not resp: return False
    # prefer value after an "answer is" cue, else last number in the response
    cue = re.search(r"answer\s*(?:is|:)?\s*([^\n.]*)", resp, re.I)
    region = cue.group(1) if cue else resp
    cands = _nums(region) or _nums(resp)
    g = _norm_num(gold)
    if atype in ("integer", "float") and g is not None:
        for c in cands:
            v = _norm_num(c)
            if v is None: continue
            if atype == "integer" and round(v) == round(g): return True
            if atype == "float":
                p = precision if isinstance(precision, int) else 2
                if round(v, p) == round(g, p): return True
        return False
    # list / text fallback: normalised substring / token match
    gs = str(gold).strip().lower().replace(",", "")
    return gs in (resp.strip().lower()) if gs else False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_sample_out")
    a = ap.parse_args()

    gold = {}
    for l in open(a.manifest):
        l = l.strip()
        if not l: continue
        r = json.loads(l); m = r.get("metadata", {}) or {}
        gold[r["id"]] = {"qt": m.get("question_type"), "ans": m.get("answer"),
                         "atype": m.get("answer_type"), "prec": m.get("precision"),
                         "letter": r.get("answer_letter")}

    tot = {"all": [0, 0], "multiple_choice": [0, 0], "free_form": [0, 0]}
    pe = 0; per = []
    for l in open(a.generations):
        l = l.strip()
        if not l: continue
        r = json.loads(l); gid = r.get("id"); g = gold.get(gid)
        if g is None: continue
        resp = r.get("response", "") or r.get("generation", "")
        qt = g["qt"] or ("multiple_choice" if g["letter"] else "free_form")
        if qt in ("multiple_choice", "multi_choice"):
            pred = first_letter(resp); ok = (pred is not None and pred == g["letter"])
            if pred is None: pe += 1
        else:
            ok = match_free(resp, g["ans"], g["atype"], g["prec"])
            if not _nums(resp) and not (resp or "").strip(): pe += 1
        tot["all"][0] += ok; tot["all"][1] += 1
        tot[qt][0] += ok; tot[qt][1] += 1
        per.append({"id": gid, "qt": qt, "correct": bool(ok), "response": resp[:300]})

    def acc(k): c, n = tot[k]; return (100.0 * c / n) if n else 0.0
    with open(a.out, "w") as f:
        w = csv.writer(f)
        w.writerow(["split", "n", "correct", "accuracy", "parse_error_count"])
        for k in ("all", "multiple_choice", "free_form"):
            w.writerow([k, tot[k][1], tot[k][0], f"{acc(k):.4f}", pe if k == "all" else ""])
    if a.per_sample_out:
        with open(a.per_sample_out, "w") as f:
            for p in per: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"MathVista acc all={acc('all'):.1f}% (n={tot['all'][1]}) | "
          f"MCQ={acc('multiple_choice'):.1f}% (n={tot['multiple_choice'][1]}) | "
          f"free_form(heuristic)={acc('free_form'):.1f}% (n={tot['free_form'][1]}) | parse_err={pe}")

if __name__ == "__main__":
    main()
