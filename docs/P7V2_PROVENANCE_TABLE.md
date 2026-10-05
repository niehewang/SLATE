# P7v2 Freeze / Overlap / Provenance Table

| Item | Recorded value | Interpretation |
|---|---|---|
| Final stage status | `STAGEP7V2_COMPLETE` | Completed |
| Held-out touched | `True` | One-shot external confirmatory executed |
| Retuned after held-out | `False` | No |
| Traffic corpus | `ShareGPT JSONL (heegyu/ShareGPT_Vicuna_unfiltered_no_imsorry)` | New external traffic domain |
| Traffic SHA256 | `fd0a9d8659486e8c1b6ec732368e4a39132adb93fb138bc70fabef834cad2076` | Frozen ShareGPT asset |
| Model family | `Phi-3-mini-4k-instruct` | New Family-B |
| Wrong source | `Qwen3-4B-Base` | Frozen enrolled negative/wrong-source reference |
| Development conversation-ID SHA | `e0ff35adbaf51b9d0b1117ed8e4bab359c76618a379c8262f9c5166e8437a970` | Deterministic development partition provenance |
| Calibration conversation-ID SHA | `a29345a6db52e91886e2ab41069613f198a2f730740017b42220e4c5a5290f91` | Deterministic calibration partition provenance |
| Held-out conversation-ID SHA | `535c3fa076ab49d89861d65c0a022378e4c4feff6141131c365e1d23533958ac` | Deterministic held-out partition provenance |
| Held-out before freeze | `NOT_JSON_PARSED_BEFORE_FREEZE` | Held-out rows not JSON-parsed before freeze |
| Dev candidate SHA | `4ef326e15ab60ae6c80010d6345a329581c1e8933eb6e234579ef06b3cbf1fb1` | Frozen development candidate pool |
| Cal candidate SHA | `809c11f065e25d1eda6efec2bcbe4d4ed856b2d21ef9f80cac8d69bec070115c` | Frozen calibration candidate pool |
| Held-out candidate SHA | `cc7a71950b68c0e8fdcaa5b934f2248354510b73e92b45f22d5573e8950654dd` | Frozen held-out candidate pool after first allowed access |
| Held-out candidate n | `128` | 128 |
| q | `64` | Frozen |
| lambda | `0.05` | Frozen before held-out |
| Similarity threshold | `0.7367470079420617` | Frozen lineage threshold |
| Risk threshold kappa | `0.9329041499981904` | Frozen risk threshold |
| Bank-gap threshold | `0.012728945276801684` | Frozen bank sufficiency threshold |
| Dev/cal Phi source bank SHA | `deaee47ff2240fd3f0ab46e319d9f070e513b879eaf1fce9847717b2b1bb655f` | Frozen response-bank provenance |
| Dev/cal Phi NF4 bank SHA | `eefc6899ddbf7451f63f206360add674c1726a9288da1fa70616a5feecee8065` | Frozen response-bank provenance |
| Dev/cal Phi LoRA bank SHA | `723cea76ada5396f2d071525665708b03a57eb91232005e57f512862d030179a` | Frozen response-bank provenance |
| Dev/cal Qwen wrong-source bank SHA | `c217e9daf739acb521b60f92903ea76f004903a0e3120dd9079b03ec2cb656f3` | Frozen response-bank provenance |
| LoRA adapter SHA | `f39941a91453b86b8b4e9c8a3db61bad9f5fce0c7caf6a3e3a59971115b71922` | LoRA frozen before ShareGPT confirmatory design |
| Dev/cal eligibility | L4=35,343/11,406; L8=6,434/2,148 | Passed before inference/freeze |
| Held-out eligibility | L4=11,836; L8=2,189 | Passed at first held-out access |

### Overlap statement

The return package records deterministic conversation-level partitions with separate conversation-ID hashes and explicitly records that held-out rows were not JSON-parsed before freeze. It does not include an explicit numeric pairwise overlap count in the returned artifacts, so the paper-facing record should claim **partition-level isolation and pre-freeze non-access**, not invent a zero-overlap count that is absent from the return package.
