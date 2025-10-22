# prepare_data.py (with progress messages)
import feedparser, requests, hashlib, random, time, re, json, sys
from bs4 import BeautifulSoup
from pathlib import Path

RSS_FEEDS = list(dict.fromkeys([
    "https://www.thairath.co.th/rss/news",
    "https://www.thairath.co.th/rss/local",
    "https://www.thairath.co.th/rss/politic",
    "https://www.thairath.co.th/rss/economy",
    "https://www.thairath.co.th/rss/sport",
    "https://www.thairath.co.th/rss/entertainment",
    "https://www.thairath.co.th/rss/foreign",
    "https://www.bangkokbiznews.com/rss/news",
    "https://mgronline.com/rss/latestnews.xml",
    "https://news.thaipbs.or.th/rss/headline.xml",
    "https://www.dailynews.co.th/rss/news",
    "https://thestandard.co/feed/",
    "https://www.matichon.co.th/feed",
    "https://www.khaosod.co.th/feed",
    "https://rssfeeds.sanook.com/rss/feeds/sanook/news.index.xml",
]))

output_file = Path("data2/t_news.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0)",
]

def clean_html(txt):
    txt = re.sub(r"<[^>]+>", " ", txt or "")
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def is_thai(text, th=0.3):
    n = len(re.findall(r"[\u0E00-\u0E7F]", text))
    return (n / max(len(text), 1)) >= th

def fetch_full(url):
    """ดึงเนื้อหาข่าวแบบเต็ม"""
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": random.choice(UA)})
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, "html.parser")

        for sel in [
            "article", "div[itemprop='articleBody']", "div.entry-content",
            "div#article-body", "section.article", "div.td-post-content",
            "div#main-content", "div.content-detail", "div.post-content",
        ]:
            el = soup.select_one(sel)
            if el:
                t = clean_html(el.get_text(" "))
                if len(t) > 200:
                    return t
        return clean_html(soup.get_text(" "))[:15000]
    except Exception:
        return ""

def text_hash(t):
    return hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()

def main():
    random.shuffle(RSS_FEEDS)
    seen_link, seen_text = set(), set()
    bag = []
    target_total = 1200
    print("📰 เริ่มโหลดข่าวจาก RSS ...\n")

    for idx, u in enumerate(RSS_FEEDS, 1):
        print(f"[{idx}/{len(RSS_FEEDS)}] 🔗 {u}")
        sys.stdout.flush()  # ให้แสดงผลทันทีใน console
        feed = feedparser.parse(u)
        if not feed.entries:
            print("   ⚠️ ไม่มีข่าวในฟีดนี้")
            continue
        got = 0
        for e in feed.entries[:200]:
            title = (e.get("title") or "").strip()
            link  = e.get("link") or ""
            if not title or not link or link in seen_link:
                continue
            desc = clean_html(e.get("description", ""))
            if len(desc) < 200:
                full = fetch_full(link)
                if len(full) > 200:
                    desc = full
            if len(desc) < 120 or not is_thai(desc):
                continue
            h = text_hash(desc)
            if h in seen_text:
                continue
            seen_link.add(link)
            seen_text.add(h)
            bag.append({
                "title": title,
                "link": link,
                "source": u,
                "text": desc
            })
            got += 1

            # แสดงความคืบหน้าทุก 10 ข่าว
            if got % 10 == 0:
                print(f"   🟢 เก็บได้ {got} ข่าวแล้ว (รวมทั้งหมด {len(bag)})")
                sys.stdout.flush()

            if len(bag) >= target_total:
                break

        print(f"   ✅ ดึงได้ {got} ข่าวจาก {u}\n")
        time.sleep(random.uniform(1.0, 1.8))
        if len(bag) >= target_total:
            break

    with output_file.open("w", encoding="utf-8") as f:
        for it in bag:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\n✅ เสร็จสิ้น! เก็บข่าวได้ทั้งหมด {len(bag)} ข่าว → {output_file}")

if __name__ == "__main__":
    main()
