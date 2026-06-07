# SPDX-License-Identifier: Apache-2.0
"""Core data model for single-image band quantification.

GUI-independent (see ``docs/adr/0001``). One :class:`Analysis` = one image
quantified for one protein, producing a raw net signal per lane. This is the
atomic unit of work: analysing one image to obtain pixel values is the "raw
data".

Normalization (loading-control ratio, condition-control ratio) combines the raw
output of *several* analyses and belongs to a separate downstream layer, not
here.

Geometry convention: boxes are axis-aligned and anchored at their top-left
corner in image pixel coordinates (numpy ``image[y, x]``). All boxes share one
global size, so they have equal area; only their positions vary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

# (x0, y0, x1, y1), half-open on the high edge.
Rect = tuple[int, int, int, int]


def overlaps(a: Rect, b: Rect) -> bool:
    """True if two axis-aligned rectangles share any area."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


class ImageRef(BaseModel):
    """Reference to the analysed image, recorded for reproducibility."""

    path: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class BoxSize(BaseModel):
    """The global, uniform ROI box size shared by every box (equal area)."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def area(self) -> int:
        return self.width * self.height


class Lane(BaseModel):
    """One gel lane.

    ``index`` is the stable key used downstream to join analyses (e.g. a target
    against its loading control) by sample. ``label`` is a human-facing name
    (a lane number or a condition name).

    ``sample`` names the *biological* sample this lane belongs to: lanes sharing
    the same ``(label, sample)`` are *technical* repeats of one sample and are
    averaged before statistics, so they do not inflate n. Lanes with the same
    ``label`` but different ``sample`` are biological replicates (the real n).
    ``included`` excludes presentation-only lanes from quantification. ``metadata``
    carries arbitrary ``item: content`` annotations (any may become a grouping
    axis later).
    """

    index: int = Field(ge=0)
    label: str
    sample: str | None = None
    included: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)


class Box(BaseModel):
    """An axis-aligned box anchored at its top-left corner (image coords)."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)

    def rect(self, size: BoxSize) -> Rect:
        return self.x, self.y, self.x + size.width, self.y + size.height


class Band(BaseModel):
    """A lane's measurement = a fixed-size box plus a same-size background box.

    Intensities are filled in by the quantification step; ``net`` is derived.
    """

    lane_index: int = Field(ge=0)
    box: Box
    background: Box
    raw: float | None = None  # sum of measurement-box pixels
    background_signal: float | None = None  # sum of background-box pixels

    @property
    def net(self) -> float | None:
        if self.raw is None or self.background_signal is None:
            return None
        return self.raw - self.background_signal


class Analysis(BaseModel):
    """One image quantified for one protein: the raw-data unit.

    Serializing an ``Analysis`` (``model_dump_json``) yields the reproducibility
    bundle: the image hash, the protein, the box size, and every box position.
    Whether this analysis is a target or a loading control is decided downstream
    when analyses are combined.
    """

    image: ImageRef
    protein: str
    expected_mw: float | None = Field(default=None, gt=0)  # kDa
    box_size: BoxSize
    lanes: list[Lane] = Field(default_factory=list)
    bands: list[Band] = Field(default_factory=list)

    def all_rects(self) -> list[Rect]:
        """Every box (measurement + background) as a rectangle."""
        rects: list[Rect] = []
        for band in self.bands:
            rects.append(band.box.rect(self.box_size))
            rects.append(band.background.rect(self.box_size))
        return rects

    @model_validator(mode="after")
    def _check_invariants(self) -> Analysis:
        # Bands must reference existing lanes.
        lane_ids = {lane.index for lane in self.lanes}
        for band in self.bands:
            if band.lane_index not in lane_ids:
                raise ValueError(f"band references unknown lane index {band.lane_index}")

        # Every box must lie within the image bounds.
        for band in self.bands:
            for box in (band.box, band.background):
                _, _, x1, y1 = box.rect(self.box_size)
                if x1 > self.image.width or y1 > self.image.height:
                    raise ValueError("box extends beyond the image bounds")

        # No box (measurement or background) may overlap another.
        rects = self.all_rects()
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if overlaps(rects[i], rects[j]):
                    raise ValueError("boxes must not overlap")

        return self
