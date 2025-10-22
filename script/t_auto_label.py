# auto_label_hf.py (fixed)
import re, json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

INPUT_FILE = Path("data/ready_for_label_soft.jsonl")
OUTPUT_FILE = Path("data/hf_labeled_news.jsonl")

MODEL_NAME = "pythainlp/thainer-corpus-v2-base-model"  # คงไว้ตามเดิม
print(f"🔹 Loading model: {MODEL_NAME}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
mdl = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

nlp_ner = pipeline(
    "ner",
    model=mdl,
    tokenizer=tok,
    aggregation_strategy="simple",  # คงไว้ แต่เราจะ dedupe ด้วย span
)

# Thai date/era ครอบคลุมขึ้น
regex_rules = {
    "DATE": re.compile(
        r"(?:\d{1,2}\s?(?:ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.|"
        r"มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
        r"(?:\s?\d{2,4})?|พ\.ศ\.\s?\d{4}|ค\.ศ\.\s?\d{4})"
    ),
    "TIME": re.compile(r"\d{1,2}:\d{2}\s?(?:น\.|am|pm|AM|PM)?"),
    "PERCENT": re.compile(r"\d+(?:\.\d+)?%"),
    "MONEY": re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:บาท|ดอลลาร์|USD|THB)"),
}

def chunk(text, max_len=350):
    sents = re.split(r'(?<=[.!?…“”\n])', text)
    cur, out = "", []
    for s in sents:
        if len(cur) + len(s) <= max_len:
            cur += s
        else:
            if cur.strip():
                out.append(cur.strip())
            cur = s
    if cur.strip():
        out.append(cur.strip())
    return out

def clean_word(w):
    w = (w or "").strip().replace("\u200b", "").replace("\xa0", "")
    if not w or re.fullmatch(r"[\(\)\[\]\-–—.,\"'«»\s]+", w):
        return None
    return w

def label_text(text):
    ents = []
    used = set()  # span-based de-dup: (start,end,label,word)

    # HF
    offset = 0
    for ch in chunk(text):
        try:
            res = nlp_ner(ch)
            for e in res:
                word = clean_word(e["word"])
                if not word:
                    continue
                # หา span ตรง ๆ ในชิ้น chunk
                for m in re.finditer(re.escape(word), ch):
                    s, t = offset + m.start(), offset + m.end()
                    key = (s, t, e["entity_group"], word)
                    if key in used:
                        continue
                    ents.append({"entity": e["entity_group"], "word": word, "score": float(e["score"]), "start": s, "end": t})
                    used.add(key)
        except Exception as ex:
            pass
        offset += len(ch)

    # Regex
    for tag, patt in regex_rules.items():
        for m in patt.finditer(text):
            s, t = m.start(), m.end()
            word = text[s:t]
            key = (s, t, tag, word)
            if key in used:
                continue
            ents.append({"entity": tag, "word": word, "score": 1.0, "start": s, "end": t})
            used.add(key)

    # คืนเป็น entities (สำคัญ: ต้องใช้คีย์นี้ให้ตรงกับสเต็ปถัดไป)
    return ents

def main():
    if not INPUT_FILE.exists():
        print("❌ missing input")
        return
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_FILE.open("r", encoding="utf-8") as fi, OUTPUT_FILE.open("w", encoding="utf-8") as fo:
        for line in fi:
            obj = json.loads(line)
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            entities = label_text(text)
            obj["entities"] = entities  # ← เปลี่ยน labels → entities ให้เข้ากับขั้นตอนถัดไป
            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"✅ wrote: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
