# script/app.py
from flask import Flask, request, render_template
from markupsafe import Markup
import requests, random, re, traceback
from bs4 import BeautifulSoup
from pathlib import Path
from pythainlp.util import normalize
import json

# --- PyThaiNLP: สรุปและตัดประโยค ---
from pythainlp.summarize import summarize
from pythainlp.tokenize import sent_tokenize

# --- HuggingFace transformers: NER ---
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# -------------------------------------------------
# ชี้โฟลเดอร์ templates = TRAIN_AI/web  ตามโครงของคุณ
# -------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../TRAIN_AI/script
ROOT_DIR = THIS_DIR.parent                           # .../TRAIN_AI
TPL_DIR  = ROOT_DIR / "web"                          # .../TRAIN_AI/web

app = Flask(__name__, template_folder=str(TPL_DIR))

# -------------------------------------------------
# ตั้งค่า HTTP headers และตัวช่วยดึง/คลีนข้อความข่าว
# -------------------------------------------------
UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0)",
]

def clean_spaces(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def fetch_full(url: str) -> str:
    """ดึงเนื้อหาข่าวจากลิงก์ พร้อม selector หลายแบบ"""
    r = requests.get(url, timeout=12, headers={"User-Agent": random.choice(UA)})
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text, "html.parser")

    selectors = [
        "article",
        "div[itemprop='articleBody']",
        "div.entry-content",
        "div#article-body",
        "section.article",
        "div.td-post-content",
        "div#main-content",
        "div.content-detail",
        "div.post-content",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            text = clean_spaces(el.get_text(" "))
            if len(text) > 200:
                return text
    return clean_spaces(soup.get_text(" "))

# -------------------------------------------------
# สรุปข่าวแบบไทย (TextRank) + สำรองกรณีล้มเหลว
# -------------------------------------------------
def summarize_th(text: str, n_sent: int = 5) -> str:
    try:
        s = summarize(text, n_sentences=n_sent)
        if s:
            return s
    except Exception:
        pass
    sents = sent_tokenize(text)
    return " ".join(sents[:n_sent])

def preprocess_for_inference(text: str) -> str:
    import re
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    text = re.sub(r"\bSpace\s*X\b", "SpaceX", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# -------------------------------------------------
# โหลดโมเดล NER (ครั้งเดียวตอนสตาร์ทแอป)
# -------------------------------------------------
NER_MODEL = "pythainlp/thainer-corpus-v2-base-model"
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

LABEL_COLOR = {
    "PERSON": "#b3d9ff",
    "ORGANIZATION": "#ffd1b3",
    "LOCATION": "#c2f0c2",
    "DATE": "#ffe680",
    "TIME": "#ffd6e7",
    "MONEY": "#e6ccff",
    "PERCENT": "#e0ffff",
    "LAW": "#ddd",
}

import unicodedata

def preprocess_for_inference(text: str) -> str:
    # รวม Space X -> SpaceX กันหลุดครึ่งตัว
    text = re.sub(r"\bSpace\s*X\b", "SpaceX", text, flags=re.I)
    # ลบอัญประกาศ/วงเล็บที่ล้อมชื่อให้เหลือแต่เนื้อ
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    text = unicodedata.normalize("NFC", text)
    # ช่องว่างซ้ำ
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

import unicodedata

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789")

def thai_to_arabic(s: str) -> str:
    return s.translate(THAI_DIGITS)

def extend_date_year_span(text: str, start: int, end: int) -> int:
    frag = text[start:end]
    # จบด้วยเลข 2 หลัก (อารบิกหรือไทย)
    two_tail = re.search(r"([0-9]{2}|[๐-๙]{2})\s*$", frag)
    if not two_tail:
        return end

    i = end
    while i < len(text) and text[i] in (" ", "\t", "\u00a0", "\u200b", ".", ",", "–", "-", "•"):
        i += 1

    # มองหาเลขอีก 2 หลักถัดไป
    m = re.match(r"([0-9]{2}|[๐-๙]{2})", text[i:])
    if m:
        return i + len(m.group(0))
    return end

STOP_ENTS = {
    "ใน","ของ","ที่","และ","หรือ","ฯ","ฯลฯ","ข่าว","สำนักข่าว","วันนี้","เมื่อวาน",
    "ผู้สื่อข่าว","รายงาน","ภาพ","คลิป"
}
STOP_DATE_WORDS = {"สิ้นเดือน","ต้นเดือน","กลางเดือน","ปลายเดือน","ต.ค."}

LABEL_COLOR = {
    "PERSON": "#b3d9ff","ORGANIZATION": "#ffd1b3","LOCATION": "#c2f0c2",
    "DATE": "#ffe680","TIME": "#ffd6e7","MONEY": "#e6ccff","PERCENT": "#e0ffff","LAW": "#ddd",
}

def normalize_ent_view(s: str) -> str:
    # ทำให้ข้อความที่โชว์สวยขึ้น (ไม่กระทบตำแหน่งในต้นฉบับ)
    s = s.strip().strip('"\''"()[] ")
    s = re.sub(r"\s{2,}", " ", s)
    return thai_to_arabic(s)

def highlight_entities(text: str):
    raw = ner(text)  # [{'start','end','word','entity_group','score'}]
    spans = []
    total_score = 0.0
    total_entity = 0  

    for r in raw:
        lab = r.get("entity_group") or r.get("entity") or "O"
        start, end = int(r.get("start", -1)), int(r.get("end", -1))
        word = (r.get("word") or "").strip()
        score = float(r.get("score", 0.0))  # ✅ ดึง score
        total_score += score
        total_entity += 1

        if lab == "O" or not word or start < 0 or end <= start:
            continue

        # ขยาย DATE
        if lab == "DATE":
            end = extend_date_year_span(text, start, end)
            word = text[start:end].strip()

        if len(word) < 2 or word in STOP_ENTS:
            continue
        if lab == "DATE" and word in STOP_DATE_WORDS:
            continue
        if re.fullmatch(r"[0-9,./:-]+", word):
            continue
        if word == "บริษัท":
            continue

        spans.append({
            "start": start,
            "end": end,
            "label": lab,
            "word": word,
            "score": score
        })

    # 🔸 กรองเอนทิตีที่ซ้อนกัน (เหมือนเดิม)
    pruned = []
    for i, a in enumerate(spans):
        contained = False
        for j, b in enumerate(spans):
            if i != j and a["label"] == b["label"] and b["start"] <= a["start"] and b["end"] >= a["end"]:
                if (b["end"] - b["start"]) > (a["end"] - a["start"]):
                    contained = True
                    break
        if not contained:
            pruned.append(a)
    spans = pruned

    # ✅ ใส่ mark
    spans.sort(key=lambda s: s["start"], reverse=True)
    html = text
    ent_table = {}

    for sp in spans:
        frag = html[sp["start"]:sp["end"]]
        color = LABEL_COLOR.get(sp["label"], "#f2f2f2")
        marked = f"<mark class='ent' style='background:{color}' title='{sp['label']} ({sp['score']:.2f})'>{frag}</mark>"
        html = html[:sp["start"]] + marked + html[sp["start"]+len(frag):]

        # ✅ เก็บเป็น dict พร้อม score
        ent_table.setdefault(sp["label"], []).append({
            "word": frag,
            "score": sp["score"]
        })
        
    avg_score = total_score / total_entity if total_entity > 0 else 0.0

    return Markup(html), ent_table, round(avg_score, 2)

# -------------------------------------------------
# Kuy Clean
# -------------------------------------------------

def clean_text(txt: str) -> str:
    """ล้าง HTML, อีโมจิ, ขยะท้ายบทความ โดยคงเครื่องหมายที่ช่วยตัดคำ"""
    if not txt:
        return ""

    # 1) เอาแท็ก HTML ออก
    txt = re.sub(r"<[^>]+>", " ", txt)

    # 2) ลบ emoji และสัญลักษณ์พิเศษนอกช่วง Unicode ไทย/อังกฤษ
    txt = re.sub(r"[\U00010000-\U0010ffff]", "", txt)

    # 3) ลบส่วนเกินที่มักเจอในเว็บข่าว
    txt = re.sub(r"พิมพ์ แชร์เรื่องนี้ แชร์เรื่องนี้ Line Twitter Facebook คัดลอกลิงก์ - ก ก", "", txt)
    txt = re.sub(r"(อ่านต่อที่.*|คลิกชมภาพ.*|ดูเพิ่มเติม.*|เครดิตภาพ.*)", "", txt)
    txt = re.sub(r"(appeared first on .*|The post .*)", "", txt, flags=re.I)
    txt = re.sub(r"(Facebook.*?Twitter.*?LINE)", "", txt)
    txt = re.sub(r"&#82\d{2};", "", txt)
    txt = re.sub(r"\s*\[\]\s*", " ", txt)
    txt = re.sub(r"อ่านข่าวต้นฉบับ:.*$", "", txt)
    txt = re.sub(r"Thairath Online Thairath Money Thairath Shopping Thairath Plus Thairath TV MIRROR หน้าแรก ฟุตบอลต่างประเทศ พรีเมียร์ลีก ยูฟ่า แชมเปียนส์ลีก บุนเดสลีกา กัลโช เซเรีย อา ลา ลีกา เจลีก ลีกอื่นๆ ตารางคะแนน", "", txt)
    txt = re.sub(r"โปรแกรม/ผลการแข่งขัน ดาวซัลโว ฟุตบอลไทย ฟุตบอลทีมชาติไทย ไทยลีก ฟุตซอลไทย Carabao 7-a-Side Cup ตารางคะแนน โปรแกรม/ผลการแข่งขัน คอลัมน์ บี บางปะกง สนามกีฬาแห่งชาติ ไฟต์สปอร์ต มวยไทย มวยโลก กีฬาโลก ", "", txt)
    txt = re.sub(r"วอลเลย์บอล แบดมินตัน มอเตอร์สปอร์ต กีฬาอื่นๆ วิดีโอ ไฮไลต์ เรื่องรอบขอบสนาม ไทยรัฐเล่ากีฬา sport daily เชียร์ไทยให้กึกก้อง แกลเลอรี่ หน้าแรก ฟุตบอลต่างประเทศ พรีเมียร์ลีก ยูฟ่า แชมเปียนส์ลีก บุนเดสลีกา กัลโช เซเรีย อา ลา ลีกา เจลีก ลีกอื่นๆ ตารางคะแนน ", "", txt)
    txt = re.sub(r"วอลเลย์บอล แบดมินตัน มอเตอร์สปอร์ต กีฬาอื่นๆ วิดีโอ ไฮไลต์ เรื่องรอบขอบสนาม ไทยรัฐเล่ากีฬา sport daily เชียร์ไทยให้กึกก้อง แกลเลอรี่ ติดตามเราได้ที่ THAIRATH ON", "", txt)
    
    # 4) normalize ตัวอักษร (เช่น พิมพ์ซ้อน / วรรณยุกต์เพี้ยน)
    txt = unicodedata.normalize("NFC", txt)
    txt = normalize(txt)

    # 5) ลบช่องว่างซ้ำ
    txt = re.sub(r"\s+", " ", txt).strip()

    # 6) คงเครื่องหมายช่วย segmentation เช่น จุด / วงเล็บ / เครื่องหมายคำพูด
    txt = re.sub(r"([?!])", r" \1 ", txt)
    txt = re.sub(r"([()\"“”‘’])", r" \1 ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    return txt

# -------------------------------------------------
# Kuy Clean V2
# -------------------------------------------------

def remove_tail_noise(text: str) -> str:
    """
    ลบข้อความส่วนท้ายที่ไม่ใช่เนื้อข่าว เช่น เครดิต อัลบั้ม แชร์เรื่องนี้ ฯลฯ
    """
    patterns = [
        r"โหลดเพิ่ม.*", r"ดูทั้งหมด\s*\d+\s*ภาพ.*", r"แชร์เรื่องนี้.*",
        r"ขอขอบคุณ.*", r"ผู้เขียน.*", r"Facebook.*", r"Twitter.*",
        r"LINE.*", r"ติดตามโซเชียล.*", r"เม้าท์กันทั้งเมือง.*"
    ]
    for p in patterns:
        text = re.split(p, text)[0]
    return text.strip()

def clean_textv2(txt: str) -> str:
    """ทำความสะอาดข้อความก่อน labeling"""
    if not txt:
        return ""
    # ลบ HTML tag และ emoji
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"[\U00010000-\U0010ffff]", "", txt)

    # Normalize ตัวอักษร (วรรณยุกต์, พิมพ์ซ้อน)
    txt = unicodedata.normalize("NFC", txt)
    txt = normalize(txt)

    # ลบเครื่องหมายคำพูดซ้ำๆ และช่องว่างเกิน
    txt = re.sub(r"[\"“”‘’]+", "", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    
    # CLEAN FUCKING STUPID WORDS
    txt = re.sub(" พิมพ์", " ", txt)
    txt = re.sub("&nbsp;", " ", txt)
    txt = re.sub(r"\s*\+\s*", " ", txt)

    txt = re.sub(r"TAGS:.*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"แหล่งอ้างอิง.*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"อ้างอิงบางส่วน:.*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"อ้างอิง .*$", "", txt, flags=re.MULTILINE)

    # ลบเครื่องหมายตกแต่ง ( - ก ก + ) ที่เจอในเว็บข่าว
    txt = re.sub(r"-\s*ก\s*ก\s*\+", "", txt)

    # ลบ timestamp, วันที่ในวงเล็บ
    txt = re.sub(r"\d{1,2}\s*[ก-ฮ]+\.\s*\d{2,4}", "", txt)
    txt = re.sub(r"\(.*?น\.\)", "", txt)

    # ลบส่วนท้ายที่ไม่ใช่เนื้อข่าว
    txt = remove_tail_noise(txt)

    # ลบ space ซ้ำอีกครั้ง
    txt = re.sub(r"\s+", " ", txt).strip()

    return txt

# -------------------------------------------------
# SAVE DATA
# -------------------------------------------------

def save_news_log(clean_text: str, summary_text: str, url_link: str):
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(logs_dir.glob("*.json"))
    next_number = len(existing_files) + 1

    filename = logs_dir / f"news_{next_number}.json"

    data = {
        "url": url_link,
        "clean_text": clean_text,
        "summary_text": summary_text
    }

    with filename.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = (request.form.get("url") or "").strip()
        n_sent = int(request.form.get("n_sent") or 5)

        if not url:
            return render_template("index.html", error="กรุณาใส่ลิงก์ข่าว")

        try:
            
            full_text = fetch_full(url)
            full_text = preprocess_for_inference(full_text) 
            full_text = clean_text(full_text)
            full_text = clean_textv2(full_text)
            if len(full_text) < 120:
                return render_template("index.html", error="ดึงเนื้อหาข่าวไม่พอ แนะนำลองลิงก์อื่น")

            summary_text = summarize_th(full_text, n_sent=n_sent)
            highlighted_html, ent_table, totalf1 = highlight_entities(summary_text)
            save_news_log(fetch_full(url), summary_text, url)
            
            f1_scores = {}
            for label in ent_table.keys():
                f1_scores[label] = 0.5
            
            return render_template(
                "result.html",
                url=url,
                summary_html=highlighted_html,
                cleantxt=fetch_full(url),
                ent_table=ent_table,
                f1_scores=f1_scores,
                totalScore=totalf1,
                full_char=len(full_text),
                sum_char=len(summary_text),
            )

        except Exception as e:
            return render_template("index.html", error=f"ประมวลผลล้มเหลว: {e}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run()