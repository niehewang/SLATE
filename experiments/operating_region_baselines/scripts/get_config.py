#!/usr/bin/env python3
import json,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'config/g4_config.json'
d=json.load(open(p,encoding='utf-8'))
print(d[sys.argv[1]])
