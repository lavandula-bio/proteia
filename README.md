# Proteia

> Interactive Western blot quantification that is fast to use, reproducible, and entirely local — from raw scan to a publication-style chart.

**Status**: Phase 1 — Western quantification MVP, in active development (pre-alpha)

Part of the [Lavandula](https://github.com/lavandula-bio) open-source ecosystem for biomedical research.

---

## What it does today

Proteia runs as a local web app: a server on your own computer does all the computation, and you use it in your browser. The server accepts connections only from that computer.

Proteia turns a Western blot scan into quantified, replicate-aware results without leaving the app:

1. **Keep each analysis as a project** — a project is a folder holding its images, lanes, proteins, and boxes; every change is saved as you make it, and Undo and Redo reach back up to 100 changes. Import TIFF, PNG, or JPEG images and say for each whether its bands are dark on a light background or light on a dark one (such as fluorescence). Proteia warns when an image is a poor basis for quantification, for example when it has lossy (JPEG-type) compression or looks like a figure prepared for display rather than a raw scan.
2. **Define the experiment** — list one condition per lane; the lane table then holds each lane's condition, biological sample (by default, each lane is its own sample), and include flag. Lanes that share both condition and sample are treated as technical repeats.
3. **Place ROIs fast** — for each protein, drag across its row of bands and Proteia places one box on each declared lane, leaving a lane without a band empty; or click a band and a region-grow step fits a box to it (Shift+click drops a fixed-size box instead). Each box stores the lane it belongs to, so a missing band leaves an empty lane instead of shifting its neighbors. All boxes of one protein share one size, which can be adjusted by hand and padded on each side, so each of its lanes is measured over the same area.
4. **Calibrate molecular weight** — choose each membrane's protein ladder from built-in presets or enter your own, click the ladder's lane and Proteia proposes a label for every ladder band, which you adjust and apply in one step; or mark the ladder bands one click at a time. A marker image taken without moving the membrane can be linked to a chemiluminescence image of the same size, so marks on one hold for both, and a second ladder on the other side of the blot lets molecular weights follow a tilted blot. The fit line reports how well each ladder fits and, with two ladders, the tilt between them.
5. **Quantify** — each band's background is measured locally, from a ring of membrane around its own box, so a gradient across the membrane does not bias a band by where it sits. A band's net signal is the sum, over its box, of how much darker each pixel is than that level (or brighter, for a light-on-dark image), with the total floored at zero. Net signals are normalized per lane to the target's loading control and, if you choose a reference condition, expressed as fold-change versus that condition.
6. **Analyze correctly** — technical repeats are averaged before statistics (so they do not inflate *n*). Proteia chooses the test from the design and states it with each chart: Student's *t*-test or one-way ANOVA + Tukey-Kramer when the conditions have equal numbers of samples, and Welch's *t*-test or Welch's ANOVA + Games-Howell when they do not. With a reference condition and three or more conditions, each condition is compared with the reference instead (Dunnett's test, or Welch's *t*-tests with Holm's correction). Ratios are tested on log values when all of them are positive.
7. **Check the results** — a Checks panel lists what could bias the numbers, among them over-exposed (clipped) bands, bands whose over-exposure could not be checked, an uneven background around a box, lanes where a band was not detected, a row box holding more bands than the protein should show, averaged technical repeats, a missing or ambiguous loading control, and, on a calibrated membrane, a ladder band whose label its neighbours contradict or two ladders that disagree.
8. **Output** — the lane table and the charts follow every edit: for each target over its loading control, a bar chart of group means with SD error bars, one point per biological sample, and significance brackets for pairs with *p* < 0.05. Export writes a new folder with the lane table as CSV (lane, condition, sample, include flag, each protein's net signal and over-exposure flag, and each target's normalized values and fold-changes), every chart as SVG and PNG, a README, and a reproducibility record: the hashes of the input images, the analysis parameters, the software version, the log of the actions taken, and the hash of every exported file.

Everything runs on your machine. **No telemetry. Your data stays local.**

## Why it exists

The ImageJ → Excel → Prism → Word path is fragmented, hard to reproduce, and easy to get wrong — and a common, quiet mistake is counting technical repeats as independent replicates. Proteia keeps quantification, normalization, and replicate-aware statistics in one reproducible flow, and is built so the analysis core stays independent of the UI.

## Roadmap

Planned, **not yet built** — listed as direction, not current capability:

- **Rows by molecular weight** — place each protein's row at its expected molecular weight along the calibrated ladders, and enter each protein's expected molecular weight on the page so its bands' apparent molecular weights are checked against it.
- **More domain rules** — further deterministic checks that flag problems in the results: pseudoreplication (technical repeats standing in for biological replicates) and loading-control checks beyond a missing or ambiguous loading control.

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
