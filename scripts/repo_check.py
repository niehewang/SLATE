#!/usr/bin/env python3
from __future__ import annotations
import pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
rc=subprocess.call([sys.executable,str(ROOT/'scripts/release_audit.py')])
if rc: sys.exit(rc)
print('REPO_CHECK_OK')
