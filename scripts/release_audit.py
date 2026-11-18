#!/usr/bin/env python3
from __future__ import annotations
import pathlib, re, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
SKIP={'scripts/release_audit.py','scripts/repo_check.py'}
TEXT_EXT={'.py','.sh','.md','.json','.txt','.yml','.yaml','.toml','.cfg','.ini','.cff','.example','.csv'}
BAD_EXT={'.pem','.key','.p12','.safetensors','.pt','.pth','.ckpt','.npz','.npy','.parquet','.arrow'}
PATS={
 'secret':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|\b(?:sk-|hf_)[A-Za-z0-9_-]{20,}|Authorization:\s*Bearer',re.I),
 'private_path':re.compile(r'(?<![A-Za-z0-9_$])/(?:home|data)/[A-Za-z0-9_.-]+/|[A-Za-z]:\\\\Users\\\\[^\\s"\']+',re.I),
 'internal_process':re.compile(r'\b(?:LATEST_HANDOFF|CURRENT_DECISION|NEXT_ACTION|PROJECT_STATE|Research OS|ChatGPT|Codex|EXPERIMENT_EXECUTION|PAPER_WRITING|IDEA_DECISION|RETURN_TO_CHATGPT)\b',re.I),
 'venue_metadata':re.compile(r'\b(?:TDSC|TPAMI|TIFS|TSC)\b|IEEE\s+Transactions\s+on\s+(?:Dependable|Computers|Information|Services)|(?:submitted|under review|targeting)\s+(?:to\s+)?IEEE',re.I),
}
errors=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts: continue
    rel=p.relative_to(ROOT).as_posix()
    if rel in SKIP: continue
    if p.suffix.lower() in BAD_EXT: errors.append(f'prohibited asset type: {rel}')
    if p.stat().st_size>50*1024*1024: errors.append(f'file >50MB: {rel}')
    if p.suffix.lower() in TEXT_EXT or p.name in {'LICENSE'}:
        txt=p.read_text(errors='ignore')
        for name,pat in PATS.items():
            if pat.search(txt): errors.append(f'{name}: {rel}')
    if re.search(r'(?:TDSC|TPAMI|TIFS|TSC|submission|reviewer|handoff|return_to_chatgpt)', rel, re.I):
        errors.append(f'public filename metadata: {rel}')
required=['README.md','LICENSE','CITATION.cff','SECURITY.md','THIRD_PARTY.md','.github/workflows/repo-check.yml']
for rel in required:
    if not (ROOT/rel).exists(): errors.append(f'missing required public file: {rel}')
if errors:
    print('PUBLIC_RELEASE_AUDIT_FAILED'); print('\n'.join(sorted(set(errors)))); sys.exit(1)
print('PUBLIC_RELEASE_AUDIT_OK')
