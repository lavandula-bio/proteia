# Proteia

> Western blot analysis from raw scan to publication-ready insight, with PhD-grade reasoning at every step.

**Status**: Pre-alpha (Phase 0)

Part of the [Lavandula](https://github.com/lavandula-bio) open-source ecosystem for biomedical research.

---

## What is this?

Proteia takes Western blot data through a complete reasoning workflow:

1. **Input** — Raw blot scan plus experimental metadata
2. **QC** — Domain-encoded rules check saturation, lane alignment, and loading control consistency
3. **Quantification** — Reproducible band detection and normalization
4. **Reasoning** — Cross-reference with protein interaction networks and relevant literature
5. **Output** — Publication-ready figure, statistics, method paragraph, and reviewer-grade interpretation, with a full audit trail

## Why this exists

Modern biomedical research generates raw data that is hours away from publishable insight. The ImageJ → Excel → Prism → Word workflow is fragmented, hard to reproduce, and easy to get wrong. Existing AI tools either reproduce errors confidently or generate plausible-sounding but unfounded interpretations.

Proteia encodes domain expertise into a deterministic and AI-augmented reasoning pipeline that:

- Catches QC errors a PhD reviewer would catch
- Generates interpretation grounded in real literature and curated knowledge databases (UniProt, Reactome, PubMed)
- Maintains a full reproducibility audit trail

## Status

**Phase 0 — Validation** (Q2-Q3 2026)

- Reasoning template design
- Baseline evaluation across LLM-only, deterministic-only, and hybrid approaches
- Architecture design for the core pipeline

**Phase 1 — Western pipeline MVP** starts Q4 2026.

## Get involved

Alpha tester recruitment opens at the start of Phase 1. We are particularly interested in:

- Researchers running regular Western blot in molecular biology, neuroscience, or related fields
- PhD students, postdocs, and analysts who feel the raw-data-to-paper friction

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
