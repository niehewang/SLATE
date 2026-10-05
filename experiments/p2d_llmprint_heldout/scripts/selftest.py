import json, pathlib, numpy as np
root=pathlib.Path(__file__).resolve().parents[1]
cfg=json.load(open(root/'config/project.json'))
assert cfg['heldout_pair_count']==64 and cfg['q']==64
assert cfg['history_lengths']==[1,4,8]
assert cfg['frozen_strongest_by_L']=={'1':'D1_L1','4':'D3_L4','8':'D3_L8'}
from run_stagep2d import flat_ids
assert flat_ids(np.array([[1,2,3]]))==[1,2,3]
print('SELFTEST_OK')
