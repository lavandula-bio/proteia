# Proteia

> Interactive Western blot quantification that is fast to use, reproducible, and entirely local — from raw scan to a publication-style chart.

**Status**: Phase 1 — Western quantification MVP, in active development (pre-alpha)

Part of the [Lavandula](https://github.com/lavandula-bio) open-source ecosystem for biomedical research.

---

## What it does today

Proteia turns a Western blot scan into quantified, replicate-aware results without leaving the app:

1. **Place ROIs fast** — click a band and a region-grow step auto-fits an equal-area box; all boxes share one size, with a matched background box for subtraction.
2. **Define the experiment** — a lane spine / data card holds each lane's condition, biological sample, and include flag. Lanes sharing a sample are treated as technical repeats.
3. **Quantify** — background-subtracted net signal per band, normalized to a loading control, expressed as fold-change versus a chosen reference condition.
4. **Analyze correctly** — technical repeats are averaged before statistics (so they do not inflate *n*), then Welch's *t*-test (2 groups) or one-way ANOVA + Tukey HSD (3+).
5. **Output** — a bar chart with individual points and significance brackets, plus CSV export of the per-lane table and chart export (PNG/PDF/SVG).

Everything runs on your machine. **No telemetry. Your data stays local.**

## Why it exists

The ImageJ → Excel → Prism → Word path is fragmented, hard to reproduce, and easy to get wrong — and a common, quiet mistake is counting technical repeats as independent replicates. Proteia keeps quantification, normalization, and replicate-aware statistics in one reproducible flow, and is built so the analysis core stays independent of the UI.

## Roadmap

Planned, **not yet built** — listed as direction, not current capability:

- **QC gates** — saturation / over-exposure detection, loading-control checks.
- **Explicit lane numbering** — click a band to set its lane, removing position-based guessing.
- **Tamper-evident reproducibility bundle** — input hash, parameters, and an auditable trail for each result.
- **Reasoning layer (experimental, optional, BYOK)** — entity extraction → curated knowledge (UniProt, Reactome, PubMed) → targeted literature → reviewer-style interpretation. This is a research direction to be validated *after* the core tool ships; it is off by default and never required to use Proteia.

## Get involved

Alpha testing is not open yet. If you run Western blots regularly (molecular biology, neuroscience, or related) and would like to hear when a testable build is ready, get in touch.

Contact: **hello@lavandula.bio**

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Lavandula™ and Proteia™ are trademarks and are not covered by the Apache License.

## About

Designed and developed by Roger Huang, a physiology PhD bridging experimental neuroscience and applied machine learning. Lavandula reflects friction points encountered on both sides of the divide.

- GitHub: [@roger79118](https://github.com/roger79118)
- LinkedIn: [Yu-Jie (Roger) Huang](https://www.linkedin.com/in/roger-huang-615925b9/)

---

*Lavandula — Distilling biomedical evidence into publishable insight.*

First use of trademarks: 2026-05-30.
