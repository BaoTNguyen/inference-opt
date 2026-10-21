"""Re-score GSM8K samples from an eval run with a final-answer extractor, for every config alike.

python3 rescore.py results/eval/<label>

lm-eval's flexible-extract takes the last number in the response, and this model ends answers with extra numbers
("**160 minutes**, or 2 hours and 40 minutes", "**$75.00** over 30 days"). This takes the first bolded number on the last
line that has one (or the last "Answer:" line) instead, falls back to the last number, and compares numerically ($26.00 == 26).
IFEval needs no re-scoring; empty answers (thinking that ran out of budget) are counted separately.
"""
import glob, json, re, sys

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def number(s):
    m = re.findall(NUM, s or "")
    return float(m[0].replace("$", "").replace(",", "")) if m else None


def final_answer(resp):
    # the last line with a bolded number states the answer; its first bolded number is the answer itself
    # ("**160 minutes**, or **2 hours and 40 minutes**" -> 160)
    for pat in (r"\*\*([^*]+)\*\*", r"[Aa]nswer\s*[:：]\s*([^\n]+)"):
        lines = [l for l in resp.splitlines() if any(re.search(r"\d", s) for s in re.findall(pat, l))]
        if lines:
            span = next(s for s in re.findall(pat, lines[-1]) if re.search(r"\d", s))
            return number(span.split("=")[-1] if re.search(r"=\s*\S*\d", span) else span)  # "**20 + 6 = 26 glasses**" -> 26
    m = re.findall(NUM, resp)
    return number(m[-1]) if m else None


def main(run_dir):
    f = sorted(glob.glob(f"{run_dir}/*/samples_gsm8k*_*.jsonl"))[-1]
    rows = [json.loads(l) for l in open(f) if '"flexible-extract"' in l]
    gold = lambda r: number(r["doc"]["answer"].split("####")[-1])
    got = [final_answer(r["resps"][0][0] if r["resps"] else "") for r in rows]
    ok = [g is not None and abs(g - gold(r)) < 1e-6 for g, r in zip(got, rows)]
    flex = sum(bool(r["exact_match"]) for r in rows)
    empty = sum(not (r["resps"][0][0] if r["resps"] else "").strip() for r in rows)
    print(f"{f}\nGSM8K: flexible-extract {flex}/{len(rows)}, final-answer {sum(ok)}/{len(rows)}, empty answers {empty}")
    for r, g, k in zip(rows, got, ok):
        if not k:
            print(f"  miss: gold {gold(r)} got {g}")
    fi = sorted(glob.glob(f"{run_dir}/*/samples_ifeval_*.jsonl"))
    if fi:
        ir = [json.loads(l) for l in open(fi[-1])]
        e = sum(not (r["resps"][0][0] if r["resps"] else "").strip() for r in ir)
        s = sum(bool(r["prompt_level_strict_acc"]) for r in ir)
        print(f"IFEval: prompt-level strict {s}/{len(ir)}, empty answers {e}, strict among answered {s}/{len(ir) - e}")


if __name__ == "__main__":
    # self-check on the formats that broke flexible-extract
    assert final_answer("So it takes **160 minutes**, or **2 hours and 40 minutes**.") == 160
    assert final_answer("He spends **$75.00** on yogurt over 30 days.") == 75
    assert final_answer("Total for the week:\n\n**20 + 6 = 26 glasses of water**") == 26
    assert final_answer("x = 5\n\n**Answer: 5 cars** drove through in the first 15 minutes.") == 5
    main(sys.argv[1])
