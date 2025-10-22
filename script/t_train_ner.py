# train_ner_thai.py (fixed)
import os, json, random, argparse
from typing import List, Tuple
import numpy as np, torch
from datasets import Dataset, load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments, Trainer)
from seqeval.metrics import classification_report, f1_score

def read_iob(path: str) -> List[List[Tuple[str,str]]]:
    sents, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur: sents.append(cur); cur=[]
                continue
            tok, lab = line.split("\t") if "\t" in line else (line, "O")
            cur.append((tok, lab))
    if cur: sents.append(cur)
    return sents

def map_labels(sents, fn):
    out=[]
    for sent in sents:
        out.append([(t, fn(l)) for t,l in sent])
    return out

def build_ds(sents):
    toks = [[t for t,_ in s] for s in sents]
    tags = [[l for _,l in s] for s in sents]
    return Dataset.from_dict({"tokens": toks, "ner_tags": tags})

def bio_type(tag): return "O" if tag=="O" or "-" not in tag else tag.split("-",1)[1]

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    true_tags, pred_tags = [], []
    for pr, lb in zip(preds, p.label_ids):
        ct, cp = [], []
        for p_i, l_i in zip(pr, lb):
            if l_i == -100: continue
            ct.append(id2label[l_i]); cp.append(id2label[p_i])
        true_tags.append(ct); pred_tags.append(cp)
    print("\n"+classification_report(true_tags, pred_tags, digits=4, zero_division=0)+"\n")
    return {"overall_f1": f1_score(true_tags, pred_tags, zero_division=0)}

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, default="data/hf_ner_dataset_iob.txt")
parser.add_argument("--eval_split", type=float, default=0.15)
parser.add_argument("--model_name", type=str, default="airesearch/wangchanberta-base-wiki-2020")
parser.add_argument("--output_dir", type=str, default="out_thai_ner")
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_length", type=int, default=320)
parser.add_argument("--label_all_tokens", action="store_true")  # เปิดด้วย flag
parser.add_argument("--drop_labels", nargs="*", default=["LEN","URL","PHONE","EMAIL"])
parser.add_argument("--focus_labels", nargs="*", default=["PERSON","LOCATION","ORGANIZATION","DATE","TIME","MONEY","PERCENT","LAW"])
args = parser.parse_args()

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# โหลด IOB ของเรา
all_sents = read_iob(args.train_file)

# ผสม ThaiNER จาก HF แบบ “ครั้งเดียว” (ไม่อ่านไฟล์ IOB ThaiNER ซ้ำ)
thainer = load_dataset("pythainlp/thainer-corpus-v2")
thainer_lbls = thainer["train"].features["ner"].feature.names
def id2name(ids): return [thainer_lbls[i] for i in ids]
for words, tags in zip(thainer["train"]["words"], thainer["train"]["ner"]):
    all_sents.append(list(zip(words, id2name(tags))))
for words, tags in zip(thainer["validation"]["words"], thainer["validation"]["ner"]):
    all_sents.append(list(zip(words, id2name(tags))))

# map labels: กรอง/โฟกัส
focus, drop = set(args.focus_labels), set(args.drop_labels)
def mapper(l):
    if l=="O" or "-" not in l: return "O"
    b, t = l.split("-",1)
    if t in drop: return "O"
    if focus and t not in focus: return "O"
    return f"{b}-{t}"

all_sents = map_labels(all_sents, mapper)
random.shuffle(all_sents)

# split
n = len(all_sents)
k = int(n*(1-args.eval_split))
train_sents, eval_sents = all_sents[:k], all_sents[k:]

# label space
labels = sorted({l for s in (train_sents+eval_sents) for _,l in s} - {""})
if "O" not in labels: labels.append("O")
labels = sorted([x for x in labels if x.startswith("B-")]) + \
         sorted([x for x in labels if x.startswith("I-")]) + ["O"]
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

tok = AutoTokenizer.from_pretrained(args.model_name)

def encode(batch):
    enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, padding=False, max_length=args.max_length)
    all_ids=[]
    for i, labs in enumerate(batch["ner_tags"]):
        wids = enc.word_ids(i)
        prev = None; lid=[]
        for wid in wids:
            if wid is None:
                lid.append(-100)
            else:
                lab = labs[wid]
                if wid != prev:
                    lid.append(label2id.get(lab, label2id["O"]))
                else:
                    if args.label_all_tokens and lab != "O":
                        typ = lab.split("-",1)[1] if "-" in lab else lab
                        lid.append(label2id.get(f"I-{typ}", label2id["O"]))
                    else:
                        lid.append(-100)
                prev = wid
        all_ids.append(lid)
    enc["labels"]=all_ids
    return enc

ds_tr = build_ds(train_sents).map(encode, batched=True)
ds_ev = build_ds(eval_sents).map(encode, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    args.model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

collator = DataCollatorForTokenClassification(tokenizer=tok)
targs = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="overall_f1",
    greater_is_better=True,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=targs,
    train_dataset=ds_tr,
    eval_dataset=ds_ev,
    data_collator=collator,
    tokenizer=tok,
    compute_metrics=compute_metrics
)

print("Labels:", labels)
trainer.train()
print(trainer.evaluate())
