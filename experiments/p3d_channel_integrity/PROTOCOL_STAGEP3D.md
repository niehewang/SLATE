# Frozen protocol

1. Verify P3B and P3C completed and embedded semantic hashes match server return artifacts.
2. Reconstruct the exact P3C heldout pool and sessions; require pool SHA match.
3. Hash-check and reuse P3C Qwen-source/NF4/Phi response banks and embedding caches.
4. Rebuild only the frozen P3B D3_L4 routing detector needed to reproduce A2 beta=.10; audit it against P3B metrics before use.
5. Freeze an evaluation-only 5% FAR consistency operating point from calibration honest sessions.
6. Evaluate the pre-existing `risk` score on heldout honest, A2, A3 exact, A3 semantic, and A4.
7. Report AUROC, TPR@5%FAR, heldout FAR, frozen-kappa TPR/FAR, verdict outcomes, bootstrap CIs, and G6.
8. No post-P3C retuning and no private-consistency rescue. If G6 fails, the module is demoted.
