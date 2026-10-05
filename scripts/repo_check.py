#!/usr/bin/env python3
from __future__ import annotations
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BAD_EXT = {'.safetensors','.pt','.pth','.ckpt','.npz','.npy','.parquet','.arrow'}
secret_patterns = [
    re.compile(r'hf_[A-Za-z0-9]{20,}'),
    re.compile(r'ghp_[A-Za-z0-9]{20,}'),
    re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
    re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
]
errors=[]
for p in ROOT.rglob('*'):
    if '.git' in p.parts or not p.is_file():
        continue
    if p.suffix.lower() in BAD_EXT:
        errors.append(f'large/binary asset should not be public: {p.relative_to(ROOT)}')
    if p.stat().st_size > 50*1024*1024:
        errors.append(f'file >50MB: {p.relative_to(ROOT)} ({p.stat().st_size} bytes)')
    if p.suffix.lower() in {'.py','.sh','.md','.json','.txt','.yml','.yaml','.toml','.cfg','.ini'}:
        txt=p.read_text(errors='ignore')
        for pat in secret_patterns:
            if pat.search(txt): errors.append(f'possible secret in {p.relative_to(ROOT)}: {pat.pattern}')
if errors:
    print('REPO_CHECK_FAILED')
    print('\n'.join(errors))
    sys.exit(1)
print('REPO_CHECK_OK')
