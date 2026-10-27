"""Build reference-start bins over transitions, excluding terminal frames."""

from __future__ import annotations

from collections.abc import Sequence


def transition_bin_layout(clip_lengths: Sequence[int], bin_frames: int = 50) -> dict:
    if isinstance(bin_frames, bool) or not isinstance(bin_frames, int) or bin_frames < 1:
        raise ValueError("bin_frames must be a positive integer")
    if not clip_lengths:
        raise ValueError("At least one clip is required")
    bin_clip, bin_start, frame_bin = [], [], []
    clip_start = 0
    for clip, length in enumerate(clip_lengths):
        if isinstance(length, bool) or not isinstance(length, int) or length < 2:
            raise ValueError("Each clip needs at least two frames to form a transition")
        terminal = clip_start + length - 1
        for start in range(clip_start, terminal, bin_frames):
            index = len(bin_start)
            bin_clip.append(clip)
            bin_start.append(start)
            frame_bin.extend([index] * (min(start + bin_frames, terminal) - start))
        # Terminal frames remain addressable for clip bookkeeping, but no bin
        # can sample them as the start of another transition.
        frame_bin.append(len(bin_start) - 1)
        clip_start += length
    return {"bin_clip": bin_clip, "bin_start": bin_start, "frame_bin": frame_bin}
