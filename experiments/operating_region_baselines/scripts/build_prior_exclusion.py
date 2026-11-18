#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def h(x): return hashlib.sha256(str(x).encode("utf-8")).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="authorized prior held-out JSONL with conversation_id and text/text_sha256")
    ap.add_argument("--output", required=True)
    args=ap.parse_args()
    conv=set(); texts=set()
    with open(args.input,encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            conv.add(h(r["conversation_id"]))
            th=r.get("text_sha256") or hashlib.sha256(str(r["text"]).strip().encode("utf-8")).hexdigest()
            texts.add(th)
    out={"format":"sha256-exclusion-v1","n_queries":len(texts),"n_conversations":len(conv),
         "conversation_id_sha256":sorted(conv),"text_sha256":sorted(texts)}
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":args.output,"n_queries":len(texts),"n_conversations":len(conv)}))
if __name__=="__main__": main()
