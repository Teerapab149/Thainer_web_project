# t_pre_clean_soft.py
import json, re, unicodedata
from pathlib import Path
from pythainlp.util import normalize

input_file = Path("data/cleaned_news.jsonl")
output_file = Path("data/ready_for_label_soft.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

# --- ฟังก์ชันหลัก ---
def soft_clean(text: str) -> str:
    if not text:
        return ""
    # normalize ตัวอักษร
    text = unicodedata.normalize("NFC", text)
    text = normalize(text)

    # ลบ HTML / emoji
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

    # ลบ pattern ที่รบกวนจริง ๆ เท่านั้น (แบบเฉพาะจุด)
    remove_patterns = [
        r"แชร์เรื่องนี้.*?(Facebook|Line|Twitter|คัดลอกลิงก์)",  # ส่วนแชร์
        r"โหลดเพิ่ม.*?อัลบั้มภาพ.*",                             # โหลดเพิ่ม
        r"ดูทั้งหมด\s*\d+\s*ภาพ.*",                                # ดูทั้งหมด xx ภาพ
        r"ขอขอบคุณ\s*ภาพ.*",                                       # เครดิตภาพ
        r"ผู้เขียน\s*[:：]?\s*[ก-ฮA-Za-z].*",                       # ผู้เขียน
    ]
    for p in remove_patterns:
        text = re.sub(p, "", text)

    # ลบ - ก ก + และพวกตกแต่ง
    text = re.sub(r"-\s*ก\s*ก\s*\+", "", text)
    # ลบช่องว่างซ้ำ
    text = re.sub(r"\s{2,}", " ", text).strip()
    # ลบ quote ซ้ำ
    text = re.sub(r"[\"“”‘’]+", "", text)

    return text.strip()


def is_valid(text: str) -> bool:
    """ยอมให้ข่าวสั้นขึ้นหน่อย เพื่อไม่ทิ้งมากเกิน"""
    if len(text) < 80:
        return False
    th_chars = len(re.findall(r"[\u0E00-\u0E7F]", text))
    if th_chars / max(len(text), 1) < 0.25:
        return False
    return True


def main():
    print("🧹 เริ่ม soft-clean ข่าว (รักษา context เดิม)...")
    total, kept = 0, 0

    with input_file.open(encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = soft_clean(item.get("text", ""))
            if is_valid(text):
                fout.write(json.dumps({
                    "title": item.get("title", "").strip(),
                    "text": text
                }, ensure_ascii=False) + "\n")
                kept += 1

            if kept % 50 == 0 and kept > 0:
                print(f"   ✅ คลีนแล้ว {kept} ข่าว (จาก {total})")

    print(f"\n🎯 เสร็จสิ้น! ข่าวพร้อม labeling ทั้งหมด {kept}/{total} → {output_file}")


if __name__ == "__main__":
    main()
