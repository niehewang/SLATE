# P7v2 Gate Summary — G2/G3/G4/G5

| Gate | Pre-specified criterion | P7v2 result | Verdict | Paper implication |
|---|---|---|:---:|---|
| G2 | Full revised SLATE lowers strongest-detector separability vs matched-unwrapped (or satisfies the frozen absolute cover criterion) | Full AUROC L1/L4/L8 = 0.5995/0.6280/0.7815; unwrapped = 0.5007/0.5066/0.6374 | **FAIL** | Do not claim cross-domain cover transfer |
| G3 | L4 BDR@AIR=.8 gain >= +0.20 vs matched-unwrapped | Full 0.6488 vs unwrapped 0.7652; gain -0.1164 | **FAIL** | Do not claim higher benign-disruption cost in Family-B/ShareGPT |
| G4 | Honest CSVR drop <= 0.05 relative to comparable baseline | Informative NF4: 1.0000 → 0.7188 (drop 0.2812); LoRA baseline=0 and is non-informative | **FAIL** | Utility preservation is engine/descendant dependent |
| G5 | At beta=.10, A1/A2 Full CSVR gain >= .15 and retention >=75% of Full honest | Aggregate A1 gain -0.3281, retention 0.4583; A2 gain -0.2812, retention 0.5833 | **FAIL** | Do not claim robust correct-source retention under bounded selective routing |

G6 was not re-tested, as required by the revised-method freeze; private consistency remains demoted.

**Overall:** P7v2 is a clean negative external confirmation. The experiment stage should close rather than opening another rescue branch.
