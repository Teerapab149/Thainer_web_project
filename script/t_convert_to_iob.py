# convert_to_iob_hf.py (fixed)
import json, re
from pathlib import Path
from pythainlp.tokenize import word_tokenize

def align_tokens_to_spans(text, tokens):
    spans, cur = [], 0
    for tok in tokens:
        m = re.search(re.escape(tok), text[cur:])
        if m:
            s = cur + m.start(); e = cur + m.end()
        else:
            s = cur; e = cur + len(tok)
        spans.append((s,e))
        cur = e
    return spans

def fix_iob(tags):
    fixed, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and (prev == "O" or prev.split("-",1)[-1] != t.split("-",1)[-1]):
            t = "B-" + t.split("-",1)[-1]
        fixed.append(t); prev = t
    return fixed

def main(inp="data/hf_labeled_news_clean.jsonl", outp="data/hf_ner_dataset_iob.txt"):
    inp, outp = Path(inp), Path(outp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with inp.open(encoding="utf-8") as fi, outp.open("w", encoding="utf-8") as fo:
        for line in fi:
            rec = json.loads(line)
            text = re.sub(r"\s+", " ", (rec.get("text") or "").strip())
            ents = rec.get("entities", [])
            tokens = [t for t in word_tokenize(text, engine="newmm") if t.strip()]
            spans = align_tokens_to_spans(text, tokens)
            labels = ["O"] * len(tokens)

            # เตรียม entity spans ทั้งหมด (start,end,label)
            es = []
            for e in ents:
                w = (e.get("word") or "").strip()
                lab = (e.get("entity") or "").strip().upper()
                if not w or not lab:
                    continue
                for m in re.finditer(re.escape(w), text):
                    es.append((m.start(), m.end(), lab))

            # ทำ labeling แบบ overlap ≥ 0.5
            for s, t, lab in es:
                touched = []
                for i, (a,b) in enumerate(spans):
                    inter = max(0, min(b, t) - max(a, s))
                    if inter > 0 and inter >= 0.5 * (b - a):
                        touched.append(i)
                if not touched:
                    continue
                labels[touched[0]] = f"B-{lab}"
                for i in touched[1:]:
                    labels[i] = f"I-{lab}"

            labels = fix_iob(labels)
            for tok, lab in zip(tokens, labels):
                fo.write(f"{tok}\t{lab}\n")
            fo.write("\n")
    print(f"✅ wrote IOB to {outp}")

if __name__ == "__main__":
    main()
