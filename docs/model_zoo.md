# Model zoo

No method has been certified yet. The zoo fills in as methods pass the gate below.

## Admission gate

A method enters only with all of: config, weights with sha256, training log, a result
record produced by the pinned evaluator, and the run snapshot.

Every number carries its provenance and they are never mixed in one column:

| Provenance | Meaning |
|---|---|
| `reported` | Taken from the paper or the authors' repository. |
| `author-ckpt` | The authors' released weights, evaluated in this toolbox. |
| `reproduced` | Trained and evaluated in this toolbox. |

Gate: `|reproduced − reported| ≤ 1.0 AP` on each of seen / similar / novel. If the gate
fails the entry may still be published, with the gap, the `author-ckpt` number and the
diagnosed cause stated. Results with and without collision filtering are always separate
rows.

## Targets

Author-reported AP on GraspNet-1Billion, RealSense, seen / similar / novel.

| Method | Collision filter | Reported | Reproduced |
|---|:---:|---|:---:|
| GSNet (ICCV 2021) | off | 65.70 / 53.75 / 23.98 | — |
| GSNet (ICCV 2021) | on | 67.12 / 54.81 / 24.31 | — |
| EconomicGrasp (ECCV 2024) | off | 68.21 / 61.19 / 25.48 | — |
| HGGD (RA-L 2023), "In paper" row of its README | unstated | 59.36 / 51.20 / 22.17 | — |
| HGGD (RA-L 2023), "In repo" row and arXiv version | unstated | 64.45 / 53.59 / 24.59 | — |
| Scale-Balanced-Grasp (CoRL 2022) | off | 58.95 / 52.97 / 22.63 | — |
| Scale-Balanced-Grasp (CoRL 2022) | on | 63.83 / 58.46 / 24.63 | — |
| FineGrasp (arXiv 2025) | off | 71.67 / 62.83 / 27.40 | — |
| FineGrasp (arXiv 2025) | on | 73.71 / 64.56 / 28.14 | — |
