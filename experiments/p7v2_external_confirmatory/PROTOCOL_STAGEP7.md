# P7v2 protocol freeze

P7v2 is an eligibility correction, not a result-driven rescue. The prior OASST traffic run stopped before detector freeze and before heldout access because it had zero valid L=4 benign user-request windows. The revised method frozen after P6 is unchanged.

Traffic corpus is the JSONL reprocessing `heegyu/ShareGPT_Vicuna_unfiltered_no_imsorry` (Apache-2.0). It preserves multi-turn conversations and permits id-only split routing before row parsing. Family-B remains Phi-3-mini-4k-instruct with NF4 and the Phi LoRA checkpoint already trained solely on OASST development traffic before this P7v2 stage.

Before model inference, the stage requires at least 512/256 development L4/L8 benign windows and 128/64 calibration L4/L8 windows. Failure stops before any response generation or heldout access. After the freeze, the one-shot heldout corpus must independently contain at least 128/64 L4/L8 benign windows; otherwise it is reported as non-evaluable without padding or cross-conversation stitching.

All lambda, detector identity, AIR/BDR thresholds, lineage/risk thresholds and scheduler parameters are frozen from ShareGPT development/calibration before heldout rows are parsed.
