# P7v2 Main Result Table — Full Revised SLATE vs Matched-Unwrapped

## Traffic cover / attacker cost

| Metric | L=1 Full | L=1 Unwrapped | L=4 Full | L=4 Unwrapped | L=8 Full | L=8 Unwrapped |
|---|---:|---:|---:|---:|---:|---:|
| Strongest-detector AUROC | 0.5995 | 0.5007 | 0.6280 | 0.5066 | 0.7815 | 0.6374 |
| BDR@AIR=.8 | 0.6422 | 0.8406 | 0.6488 | 0.7652 | 0.3897 | 0.5861 |
| AIR@BDR=.1 | 0.1836 | 0.1094 | 0.1977 | 0.1148 | 0.3761 | 0.1579 |

## Source-verification outcome

| Descendant | Serving | Full CSVR | Unwrapped CSVR | Full FSVR | Full Channel-Risk | Full Insufficient | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Phi NF4 | Honest | 0.7188 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | Informative baseline; Full loses utility |
| Phi NF4 | A1 | 0.3438 | 1.0000 | 0.0000 | 0.0000 | 0.0312 | No defense gain |
| Phi NF4 | A2 | 0.4375 | 1.0000 | 0.0312 | 0.0000 | 0.0000 | No defense gain |
| Phi LoRA | Honest | 0.0312 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Baseline ineligible (CSVR=0) |
| Phi LoRA | A1 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | Abstention only |
| Phi LoRA | A2 | 0.0000 | 0.0000 | 0.0000 | 0.8438 | 0.8438 | Abstention only |
