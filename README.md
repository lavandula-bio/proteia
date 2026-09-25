# Proteia

> Interactive Western blot quantification that is fast to use, reproducible, and entirely local — from raw scan to a publication-style chart.

**Status**: Phase 1 — Western quantification MVP, in active development (pre-alpha)

Part of the [Lavandula](https://github.com/lavandula-bio) open-source ecosystem for biomedical research.

---

## What it does today

Proteia turns a Western blot scan into quantified, replicate-aware results without leaving the app:

1. **Place ROIs fast** — while editing a protein, Ctrl+click a band and a region-grow step fits a box to it (in manual-box mode, a fixed-size box is dropped on the click instead); all boxes for one protein share one size, which starts from the largest fitted band and can be adjusted by hand, so each of its lanes is measured over the same area.
2. **Define the experiment** — list one condition per lane; a data card then holds each lane's condition, biological sample (by default, each lane is its own sample), and include flag. Lanes that share both condition and sample are treated as technical repeats. Boxes are matched to lanes by their horizontal position within each image, which can misplace lanes when a band is missing; Proteia warns when a protein has fewer boxes than lanes.
3. **Quantify** — each image gets a single background level, the median of its grayscale pixels (there is no separate background box per band). A band's net signal is the sum, over its box, of how much darker each pixel is than that level (or brighter, for an image you set as light-on-dark, such as fluorescence); pixels on the background side count as zero. Net signals are normalized per lane to a loading control and, if you choose a reference condition, expressed as fold-change versus that condition.
4. **Analyze correctly** — technical repeats are averaged before statistics (so they do not inflate *n*), then Welch's *t*-test (2 groups) or one-way ANOVA + Tukey HSD (3+).
5. **Output** — a bar chart of group means with SD or SEM error bars, one point per biological sample, and significance brackets for pairs with *p* < 0.05, plus CSV export of the per-lane table (lane, condition, sample, include flag, and each protein's net signal) and chart export (PNG/PDF/SVG).

Everything runs on your machine. **No telemetry. Your data stays local.**

## Why it exists

The ImageJ → Excel → Prism → Word path is fragmented, hard to reproduce, and easy to get wrong — and a common, quiet mistake is counting technical repeats as independent replicates. Proteia keeps quantification, normalization, and replicate-aware statistics in one reproducible flow, and is built so the analysis core stays independent of the UI.

## Roadmap

Planned, **not yet built** — listed as direction, not current capability:

- **Local web app** — use Proteia in your browser while a local server on your own computer does all the computation; the server accepts connections only from that computer, and the app works offline. It replaces the current napari-based desktop interface.
- **Project files** — save an analysis as a project folder (images, lane table, proteins, boxes, and exports) and reopen it later to continue or review it.
- **Reproducibility record** — every project and export carries the hashes of the input images, the analysis parameters, the software version, and a log of the actions taken, so each result can be traced back to how it was produced.
- **Explicit lane identity** — each band stores the declared lane it belongs to, so a missing band leaves an empty slot instead of shifting its neighbors; horizontal position only proposes a lane.
- **Row-based band detection** — drag one box around a row of bands; Proteia finds one band per declared lane, places an equal-size box on each, and leaves lanes without a band empty. Clicking individual bands remains available as a fallback.
- **Molecular-weight guidance** — calibrate each membrane against its protein ladder (for example, from a marker image taken without moving the membrane) and place each protein's row at its expected molecular weight.
- **Domain rules** — deterministic checks that flag problems in the results: over-exposed (clipped) bands, molecular-weight consistency (apparent versus expected), band count (found versus expected), pseudoreplication (technical repeats standing in for biological replicates), and loading-control checks.

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
