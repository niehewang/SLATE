# P7v2 — Revised SLATE ShareGPT + Phi external confirmatory

This stage supersedes the aborted OASST traffic attempt **before any freeze or heldout access**. OASST1 lacked valid L=4 benign user-request windows under the frozen user-side history definition. P7v2 therefore uses a JSONL ShareGPT multi-turn traffic corpus while reusing the already-completed Phi OASST-development LoRA descendant.

## Scientific safety
- OASST attempt: `heldout_touched=false`, no detector freeze existed.
- P7v2 traffic: `heegyu/ShareGPT_Vicuna_unfiltered_no_imsorry`, Apache-2.0, JSONL.
- Exact traffic file SHA256: `fd0a9d8659486e8c1b6ec732368e4a39132adb93fb138bc70fabef834cad2076`.
- Before *any* model inference, development/calibration must have sufficient L=4 and L=8 benign windows. After the frozen manifest, heldout is opened once and must independently satisfy the predeclared L4/L8 evaluability floor; failure is reported as confirmatory-traffic ineligibility, never padded or cross-session stitched.
- Heldout JSONL rows are skipped by id before `json.loads` until the freeze manifest exists.
- No official Hugging Face or GitHub URL is used; `RUN_THIS.sh` is mirror-only.

## Run
```bash
bash SELFTEST.sh
bash RUN_THIS.sh
```

Return: `/data/jx-vmlab/SLATE_SERVER_RETURN_STAGEP7V2_REVISED_SHAREGPT_PHI_EXTERNAL_CONFIRMATORY_V1.tar.gz`
