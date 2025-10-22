# clean_labeled_news_hf.py (fixed)
import json, re, unicodedata
from pathlib import Path
from pythainlp.util import normalize

INPUT = Path("data/hf_labeled_news.jsonl")
OUTPUT = Path("data/hf_labeled_news_clean.jsonl")

VALID_LABELS = {"PERSON","LOCATION","ORGANIZATION","DATE","TIME","MONEY","PERCENT","LAW"}
STOPWORDS = {"ของ","ที่","ใน","โดย","กับ","เป็น","เมื่อ","ได้","จะ","และ","หรือ","จาก","ถึง"}
THRESHOLD = {
    "PERSON": 0.80, "LOCATION": 0.80, "ORGANIZATION": 0.80,
    "DATE": 0.70, "TIME": 0.70, "MONEY": 0.70, "PERCENT": 0.70, "LAW": 0.70
}
FAKE_NAMES = {"รัฐบาล","นายกรัฐมนตรี","รัฐมนตรี","ผู้ว่าฯ","ผู้กำกับการ","คณะกรรมการ","ตำรวจ"}

def clean_text(text):
    text = re.sub(r"\[&#.*?;\]", "", text)
    text = re.sub(r"appeared first on .*", "", text, flags=re.I)
    text = re.sub(r"The post .*", "", text, flags=re.I)
    text = unicodedata.normalize("NFC", text)
    text = normalize(text)
    text = re.sub(r"\s+", " ", text.strip())
    return text

def clean_entities(entities):
    cleaned = []
    for e in entities or []:
        label = (e.get("entity") or "").strip().upper()
        word  = (e.get("word") or "").strip()
        score = float(e.get("score", 1.0))
        if not word or label not in VALID_LABELS:
            continue
        if word in STOPWORDS:
            continue
        if word in FAKE_NAMES and label == "PERSON":
            # ยอมให้เป็น ORG/ORG-type ได้มากกว่า PERSON
            label = "ORGANIZATION"
        if score < THRESHOLD.get(label, 0.7):
            continue
        if len(word) <= 1 and label not in {"DATE","TIME","LAW"}:
            continue
        word = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", "", word)
        word = re.sub(r"\s+", " ", word).strip()
        if not word:
            continue
        cleaned.append({"entity": label, "word": word, "score": round(score,3)})
    return cleaned

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    n_total = n_clean = 0
    with INPUT.open(encoding="utf-8") as f, OUTPUT.open("w", encoding="utf-8") as w:
        for line in f:
            rec = json.loads(line)
            n_total += 1
            text = clean_text(rec.get("text",""))
            ents = clean_entities(rec.get("entities", []))
            if not text:
                continue
            json.dump({"text": text, "entities": ents}, w, ensure_ascii=False)
            w.write("\n")
            n_clean += 1
    print(f"✅ Cleaned {n_clean}/{n_total} → {OUTPUT}")

if __name__ == "__main__":
    main()
