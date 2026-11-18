#!/usr/bin/env python3
import json,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'config/g3_config.json'; d=json.load(open(p))
print(d[sys.argv[1]])
