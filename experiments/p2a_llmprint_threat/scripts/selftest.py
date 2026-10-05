import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cfg=json.load(open(root/'config/project.json'))
assert cfg['pair_count_total']==128
assert cfg['train_pairs']==64 and cfg['eval_pairs']==64
assert cfg['history_lengths']==[1,4,8]
assert cfg['benign_window_cap']==7000
print('SELFTEST_OK')
