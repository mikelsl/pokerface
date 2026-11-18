# pokerface — archived research prototype

**Status: stopped / documentation only.** This repository candidate records an offline, CPU-oriented study of whether pre-bet face/hand features in televised poker have measurable association with a *showdown-strength proxy*. It is **not a lie detector**, does not infer a player's intent, and has **no demonstrated predictive value**. Do not deploy it for real-time or consequential judgments.

The owner paused source collection, census, and further training on 2026-11-06 and is considering closing the project. This archive does not resume collection, downloading, retraining or model deployment. The sole real-data diagnostic run was on 2026-11-05; results and limitations: [docs/RESULTS.zh-CN.md](docs/RESULTS.zh-CN.md). Decisions and methods: [docs/DECISIONS.zh-CN.md](docs/DECISIONS.zh-CN.md).

## Scope of the snapshot

- `code/source_baseline/poker-river-research/riverlab/` (v0 baseline): Python modules; independent card comparison, media masking, frozen-feature extraction and synthetic four-way evaluation.
- `v2/code/contract.py` (v2 contract): frozen 5-second pre-bet window, multiway showdown-strength proxy, fail-closed feature provenance checks.
- `v3_train/train_v2.py` (v3 trainer): preregistered primary/diagnostic protocol, session-disjoint splits, synthetic smoke path and tests; includes a separate label-blind feature-source audit script.
- v3_train/PREREGISTRATION.md: public name-redacted copy of the historical preregistration; historical hashes refer to withheld private originals. Synthetic results are not real-world performance.

**Excluded on purpose:** broadcast video/audio, frames/screenshots, masked clips, raw hands, feature rows, labels, trained model bundles, MediaPipe model weights, cookies, credentials, local review archives, download code, queues and unreviewed census outputs. No copyright/license grant for third-party video or MediaPipe models is implied. A source URL is not a license to redistribute footage.

## Reproduction boundary

The directory layout matches the original local layout to make imports explicit. Python dependencies are described in `code/source_baseline/poker-river-research/requirements.txt`; this candidate does **not** contain the real rows or video needed to reproduce the historical real-data diagnostic. A synthetic-only software smoke test can be run from the root after installing compatible dependencies:

```bash
python v3_train/train_v2.py --synthetic --outdir v3_train/synthetic_release_smoke
```

The output is **synthetic only** and must never be cited as a poker accuracy result. Some original tests intentionally require withheld local real data and will not pass in this public-data-free snapshot. `v3_train/train_v2.py` requires both `--real` and `--approved-by-owner` before real-data fitting; that historical command-line guard is not a substitute for consent, licensing or research ethics.

## Publication status

The owner authorized a public pokerface repository, no personal attribution, and unrestricted use of the original code and documentation under the Unlicense (see LICENSE). Third-party dependencies, MediaPipe models and broadcast footage are not covered. The reviewed snapshot was published by SSH as a fast-forward descendant of the initial placeholder README; see docs/VERIFICATION.zh-CN.md for the scope and limits of verification.

中文说明：[决策过程](docs/DECISIONS.zh-CN.md) · [研究进展与结果](docs/RESULTS.zh-CN.md) · [本机验证](docs/VERIFICATION.zh-CN.md) · [发布审查](docs/PUBLISH_REVIEW.zh-CN.md).
