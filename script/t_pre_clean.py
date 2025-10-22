# t_pre_clean_v2.py
import json, re, unicodedata
from pathlib import Path
from pythainlp.util import normalize

# ---------- PATH ----------
input_file = Path("data/t_news.jsonl")   # ไฟล์จากขั้นตอนก่อนหน้า
output_file = Path("data/cleaned_news.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

# ---------- ฟังก์ชันช่วย ----------
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

def clean_text(txt: str) -> str:
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

def is_valid(text: str) -> bool:
    """เช็กว่าข่าวเหมาะสมสำหรับ labeling หรือไม่"""
    if len(text) < 100:
        return False
    th_chars = len(re.findall(r"[\u0E00-\u0E7F]", text))
    if th_chars / max(len(text), 1) < 0.4:
        return False
    return True


# ---------- MAIN ----------
def main():
    print("🧼 เริ่มทำความสะอาดขั้นสุดท้าย เพื่อเตรียม labeling ...")
    total, kept = 0, 0

    with input_file.open(encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = clean_text(item.get("text", ""))
            if is_valid(text):
                fout.write(json.dumps({
                    "title": item.get("title", "").strip(),
                    "text": text
                }, ensure_ascii=False) + "\n")
                kept += 1

            if kept % 50 == 0 and kept > 0:
                print(f"   ✅ คลีนแล้ว {kept} ข่าว (จาก {total})")

    print(f"\n🎯 เสร็จสิ้น! เตรียมข่าวพร้อม labeling ได้ทั้งหมด {kept}/{total} ข่าว → {output_file}")


if __name__ == "__main__":
    main()