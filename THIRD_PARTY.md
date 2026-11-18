# Third-party components

SLATE wraps or evaluates external evidence/threat baselines and uses public model/data assets. The MIT license at the repository root applies only to original SLATE repository code and documentation; it does **not** relicense third-party software, model weights, or datasets.

## LLMPrint

Used in the P2 threat-characterization chain. This repository contains SLATE-side orchestration/adaptation scripts and frozen metadata. Official LLMPrint artifacts/source remain governed by the upstream project's own terms.

## LLM-DNA / RepTrace

Used as a lineage evidence engine in P3/P5/P6/P7. The frozen experiments reference LLM-DNA/RepTrace 1.0.1 and the exact release/commit information recorded in the stage manifests. Upstream source and license obligations remain with that project.

## Models / encoders

The experiment chain references external model families and encoders, including Qwen, Phi and sentence-embedding models. Checkpoints are not redistributed here and remain subject to their upstream licenses and terms.

## Datasets

WildChat, ShareGPT, OASST-derived assets and other traffic/query corpora are not redistributed in this repository. Obtain them independently from their upstream source or an authorized mirror and follow the applicable license/terms.

## Redistribution rule

Do not add third-party repositories, model weights, datasets, or generated assets to this repository merely because they were used in an experiment. If redistribution is ever required, preserve the upstream copyright/license notices and document provenance explicitly.
