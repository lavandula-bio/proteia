# SPDX-License-Identifier: Apache-2.0
"""The channel rule (#206): which channel a colour image is measured on, and
the words that say why.

Pure: it holds the rule's constants, decides from the band masses
:func:`~proteia.core.imaging.channel_masses` measures, and words the choice
for the page and the results, so both read one source. It reads no pixel and
no file.

Version 1, on the pixels as read (alpha dropped), at the image's polarity,
with the constants of :data:`VERSIONS`:

1. Each channel's values turned signal-up: the pixel for light bands on a dark
   background, the detector's top minus the pixel for dark bands on a light one.
2. Its median and ``sigma``, ``mad_sigma`` times the median absolute deviation
   but at least ``min_sigma``, both exact from a histogram of the integer
   values; its cut is the median plus ``cut_sigmas`` sigmas.
3. Its regions: 8-connected pixels above the cut, kept where they hold
   ``min_region`` pixels or more (a smaller one is dust or noise).
4. Its band mass: the sum over its kept pixels of their height above the cut,
   divided by the image's pixel count (grey levels per pixel).
5. No channel with a mass above 0: the mean, "no channel held bands above its
   background".
6. Every channel's mass at least ``all_carry`` of the largest (the all-carry
   threshold): the mean, "all three channels hold its bands" (a gray blot saved
   in colour, an imager's overlay), as ImageJ measures colour by default.
7. Else the channel with the largest mass, in the order red, green, blue on a
   tie.
8. Apart from it (``apart``): each other channel's mass in kept regions more
   than ``apart_gap`` pixels from the picked channel's kept regions, as a share
   of the picked channel's mass; 0 for the pick and for a mean pick. A share of
   ``apart_share`` or more names that channel among the image's signal
   channels (:func:`signal_channels`): another protein, maybe, which the
   image's own channel does not measure.

A stored choice keeps the version of the rule that made it (``rule``) and the
masses it was made on, and is read with that version's constants: a later
build that changes a constant adds a version, measures new choices with it,
and changes no stored choice or its words, which are worked out again from
the stored values alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, NamedTuple

from pydantic import JsonValue

from proteia.core.model import Channel, ChannelChoice, ChannelSource, in_words


class Constants(NamedTuple):
    """The constants of one version of the rule (the steps above)."""

    cut_sigmas: float
    mad_sigma: float  # a normal distribution's sigma per median absolute deviation
    min_sigma: float  # grey levels: a channel of nearly constant values still has a cut
    min_region: int  # pixels, 8-connected
    # Step 6, the all-carry threshold: the mean once the weakest channel holds
    # this share of the largest mass or more. Above 1, never: every image whose
    # bands lie in some channel is measured on its strongest one.
    all_carry: float
    apart_share: float
    apart_gap: int  # pixels, 8-connected


# Every version of the rule this build reads, by number. A change to any
# constant is a new version, measured with from then on (VERSION).
VERSIONS: Final[dict[int, Constants]] = {
    1: Constants(
        cut_sigmas=3.0,
        mad_sigma=1.4826,
        min_sigma=0.5,
        min_region=16,
        all_carry=0.5,
        apart_share=0.1,
        apart_gap=2,
    ),
}
VERSION: Final = 1  # the version this build measures with
COLOURS: Final = (Channel.RED, Channel.GREEN, Channel.BLUE)


def constants(version: int | None = None) -> Constants:
    """The constants of the rule's ``version`` (None: the one this build
    measures with, :data:`VERSION`); ``KeyError`` for one it does not know."""
    return VERSIONS[VERSION if version is None else version]


# The channel rule's words, shared by the page's line under "Measured on"
# (:func:`line`), the results' colour_channel note (:func:`note`) and the
# next step of their clipping_not_checked notice (:data:`TO_CHECK`).
_MOST: Final = "holds the most signal above its background"
_ALL_HOLD: Final = "all three channels hold its bands"
_NONE_HELD: Final = "no channel held bands above its background"
_IMAGEJ: Final = "To compare with ImageJ, measure the same channel (Image > Color > Split Channels)"
_BY_DEFAULT: Final = "as ImageJ measures colour by default"
_HOLDS_BANDS: Final = "the channel that holds its bands"
_TO_CHOOSE: Final = f"choose {_HOLDS_BANDS} to measure it there (its boxes are measured again)."
# The page's line for an image no channel was chosen or picked for: one
# imported before Proteia picked a channel.
_UNCHOSEN_LINE: Final = (
    f"No channel chosen: measured on the mean of its red, green and blue. C{_TO_CHOOSE[1:]}"
)
# What to check first where no channel held bands: most often the polarity.
_POLARITY: Final = "its polarity (Bands are"
# How to choose the channel to check over-exposure on exactly (the next step of
# clipping_not_checked on a lossless colour image measured on the mean). On an
# export whose imager marked its saturated pixels in a colour, every channel
# holds the bands, and in two of them a mark reads as a hole in the band, which
# hides its saturation: the step says which channel keeps them.
TO_CHECK: Final = (
    f"choose {_HOLDS_BANDS} for its image to check it there (if the imager marked saturated"
    " pixels in a colour, the channel in which those marks look like the strongest part of"
    " the band)"
)
# What the channel_signal_elsewhere warning tells the user to do with a
# channel that holds signal apart from the image's: measure it on a copy of
# the image, on that channel, in the same project. For one channel, then for
# several.
_ANOTHER_COPY: Final = (
    "if that is another protein, measure it on its own: import the same file again and"
    " choose {channel} for the new image"
)
_OTHER_COPIES: Final = (
    "if those are other proteins, measure each on its own: import the same file again for"
    " each and choose its channel for the new image"
)


def settings() -> dict[str, JsonValue]:
    """The rule as a reproducibility record states it
    (:func:`~proteia.core.record.settings`): its version, constants and steps.
    Each image's choice, with the version and the masses it was made on, is in
    the record's content."""
    rule = constants()
    return {
        "rule": VERSION,
        "picks_for": (
            "a colour image whose red, green and blue differ, on import and on a polarity"
            " change; not a visible-light marker image, nor an image converted from CMYK"
        ),
        "values": (
            "each of red, green and blue as read: the pixel for light bands on a dark"
            " background, the detector's top minus the pixel for dark bands on a light one"
        ),
        "cut": "median + cut_sigmas * max(mad_sigma * median absolute deviation, min_sigma)",
        "cut_sigmas": rule.cut_sigmas,
        "mad_sigma": rule.mad_sigma,
        "min_sigma": rule.min_sigma,
        "regions": "8-connected pixels above the cut, of min_region_pixels pixels or more",
        "min_region_pixels": rule.min_region,
        "mass": "the regions' summed height above the cut, per pixel of the image",
        "mean": (
            "when no channel has a mass above 0, or every channel's mass is at least"
            " all_carry of the largest"
        ),
        "all_carry": rule.all_carry,
        "else": "the channel of the largest mass; on a tie, the first of red, green, blue",
        "apart": (
            "each other channel's mass in regions more than apart_gap_pixels (8-connected)"
            " from the pick's regions, as a share of the pick's mass; a share of apart_share"
            " or more is signal the image's channel does not measure"
        ),
        "apart_gap_pixels": rule.apart_gap,
        "apart_share": rule.apart_share,
    }


def pick(masses: Sequence[float], version: int | None = None) -> Channel:
    """The channel the rule's ``version`` (None: :data:`VERSION`) picks from
    the band masses of red, green and blue (steps 5 to 7): the mean if none is
    above 0 or all carry the bands, else the largest, the first of red, green,
    blue on a tie."""
    best = max(masses)
    if best <= 0:
        return Channel.MEAN
    if all(mass >= constants(version).all_carry * best for mass in masses):
        return Channel.MEAN
    return COLOURS[list(masses).index(best)]


def signal_channels(choice: ChannelChoice | None) -> tuple[Channel, ...]:
    """The channels of an image that hold signal, from the evidence its choice
    stores (step 8), read with the constants of the version that measured it,
    in the order red, green, blue: that version's pick when it is one channel,
    and every channel whose mass apart from it is at least its ``apart_share``
    of it. Empty without evidence (no choice, or one stored before the rule
    measured any), and for a version this build does not know (a later
    build's), rather than a guess."""
    if choice is None or choice.rule is None or choice.masses is None or choice.apart is None:
        return ()
    rule = VERSIONS.get(choice.rule)
    if rule is None:
        return ()
    picked = pick(choice.masses, choice.rule)
    return tuple(
        colour
        for colour, apart in zip(COLOURS, choice.apart, strict=True)
        if colour is picked or apart >= rule.apart_share
    )


def _percent(share: float) -> str:
    """A share as a whole percentage, rounded half up; never 100% short of all."""
    whole = int(share * 100 + 0.5)
    return f"{min(whole, 99) if share < 1 else whole}%"


def _shares(choice: ChannelChoice) -> str:
    """The other channels' masses as shares of the picked one's: ``red 3%,
    blue 2%``."""
    masses = choice.masses
    assert masses is not None  # a picked choice holds its masses (the model checks)
    best = max(masses)
    return ", ".join(
        f"{colour} {_percent(mass / best)}"
        for colour, mass in zip(COLOURS, masses, strict=True)
        if colour is not choice.channel
    )


def line(choice: ChannelChoice | None) -> str:
    """The page's line under "Measured on" for an image with this choice: who
    made it and why."""
    if choice is None:
        return _UNCHOSEN_LINE
    if choice.source is ChannelSource.CHOSEN:
        return "Chosen by you."
    if choice.channel is not Channel.MEAN:
        return f"Picked by Proteia: it {_MOST} ({_shares(choice)} of it)."
    if _no_regions(choice):
        return f"Picked by Proteia: {_NONE_HELD}. Check {_POLARITY}), or {_TO_CHOOSE}"
    return f"Picked by Proteia: {_ALL_HOLD}."


def _no_regions(choice: ChannelChoice) -> bool:
    return choice.masses is not None and max(choice.masses) <= 0


def note(choice: ChannelChoice | None, image: str) -> str:
    """The ``colour_channel`` note of an image named ``image`` with this
    choice: what it is measured on, who chose it and why, and how ImageJ
    measures the same."""
    if choice is None:
        return (
            f"{image} is measured on the mean of its red, green and blue, as Proteia measured"
            " colour images before it chose a channel; choose a channel for it under Images to"
            " measure it there (its boxes are measured again)"
        )
    channel = choice.channel
    if choice.source is ChannelSource.CHOSEN:
        if channel is Channel.MEAN:
            return (
                f"{image} is measured on the mean of its red, green and blue, as chosen (ImageJ"
                " measures colour on the mean by default)"
            )
        return f"{image} is measured on its {channel} channel, as chosen. {_IMAGEJ}"
    if channel is not Channel.MEAN:
        return (
            f"{image} is measured on its {channel} channel, which Proteia picked: it {_MOST} of"
            f" the three. {_IMAGEJ}"
        )
    if _no_regions(choice):
        return (
            f"{image} is measured on the mean of its red, green and blue: {_NONE_HELD} when"
            f" Proteia picked its channel; check that {_POLARITY}, under Images) matches its"
            " bands, or, if its bands are of one colour, choose that channel for it under Images"
        )
    return f"{image} is measured on the mean of its red, green and blue, {_BY_DEFAULT}: {_ALL_HOLD}"


def elsewhere(image: str, channels: Sequence[Channel]) -> str:
    """The ``channel_signal_elsewhere`` warning of an image named ``image``
    whose ``channels`` (one or more, in order) hold signal apart from the
    channel it is measured on, which no image of its file measures."""
    names = in_words([str(channel) for channel in channels])
    if len(channels) == 1:
        where, step = f"{names} channel", _ANOTHER_COPY.format(channel=names)
    else:
        where, step = f"{names} channels", _OTHER_COPIES
    return f"{image} holds signal in its {where} apart from the bands it is measured on: {step}"
