"""Generate MPEG-TS filler (black frames + silence) via FFmpeg for ad replacement."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

from streamlink.logger import getLogger


log = getLogger(__name__)

# Target filler bitrate used to calculate how many bytes to read for a given duration.
# Measured: libx264 ultrafast+stillimage with black/solid frames + AAC silence produces
# ~11-12 KB/s regardless of resolution.  15 KB/s provides margin for complex images.
_FILLER_BYTES_PER_SECOND = 15_000

# MPEG-TS constants
_TS_PACKET_SIZE = 188
_TS_SYNC_BYTE = 0x47

# 90 kHz clock used for MPEG-TS PTS/DTS timestamps
_TS_CLOCK_RATE = 90_000

# Fixed encoder priming delay: libx264 ultrafast + AAC in mpegts consistently
# produces first output at ~1.4s regardless of resolution.  We subtract this
# from the requested ts_offset so the first filler frame lands at the right PTS.
_ENCODER_STARTUP_DELAY = 1.4


def extract_last_pts(data: bytes) -> int | None:
    """Extract the highest PTS value from MPEG-TS data.

    Scans all TS packets for PES headers and returns the largest PTS found,
    as a raw 33-bit value in 90 kHz ticks.  Returns None if no PTS is found.
    """

    best_pts: int | None = None

    for offset in range(0, len(data) - _TS_PACKET_SIZE + 1, _TS_PACKET_SIZE):
        if data[offset] != _TS_SYNC_BYTE:
            continue

        # Check adaptation field control and payload start indicator
        has_payload = bool(data[offset + 3] & 0x10)
        payload_start = bool(data[offset + 1] & 0x40)
        if not has_payload or not payload_start:
            continue

        # Skip adaptation field if present
        adaptation = (data[offset + 3] >> 4) & 0x03
        if adaptation == 0x03:
            af_len = data[offset + 4]
            payload_offset = offset + 5 + af_len
        elif adaptation == 0x01:
            payload_offset = offset + 4
        else:
            continue

        # Check for PES start code (0x000001) and sufficient room for PES header
        end = offset + _TS_PACKET_SIZE
        if payload_offset + 9 > end:
            continue
        if data[payload_offset] != 0 or data[payload_offset + 1] != 0 or data[payload_offset + 2] != 1:
            continue

        # PES header: check PTS_DTS flags in byte 7
        pts_dts_flags = (data[payload_offset + 7] >> 6) & 0x03
        if pts_dts_flags < 2:
            continue

        pts_start = payload_offset + 9
        if pts_start + 5 > end:
            continue

        pts = _read_pts(data, pts_start)
        if best_pts is None or pts > best_pts:
            best_pts = pts

    return best_pts


def _read_pts(data: bytes | bytearray, offset: int) -> int:
    """Read a 33-bit PTS/DTS value from 5 bytes at the given offset."""

    b0, b1, b2, b3, b4 = data[offset], data[offset + 1], data[offset + 2], data[offset + 3], data[offset + 4]
    return (
        ((b0 >> 1) & 0x07) << 30
        | b1 << 22
        | ((b2 >> 1) & 0x7F) << 15
        | b3 << 7
        | (b4 >> 1) & 0x7F
    )


class FFmpegFiller:
    """FFmpeg-based MPEG-TS filler generator for ad replacement.

    Produces black frames (or a user-supplied image) plus silent audio
    via lavfi sources, encoded to match the stream's resolution/framerate
    so downstream tools never see a format change mid-file.

    A new FFmpeg process is started per ad break via ``start(ts_offset=…)``
    so that output timestamps are contiguous with the real stream.
    """

    def __init__(
        self,
        ffmpeg_path: str,
        width: int,
        height: int,
        framerate: float,
        image: str | Path | None = None,
    ):
        self._ffmpeg_path = ffmpeg_path
        self._width = width
        self._height = height
        self._framerate = framerate
        self._image = str(image) if image else None
        self._process: subprocess.Popen | None = None

    # -- public API ------------------------------------------------------------

    def start(self, ts_offset: float | None = None) -> None:
        """Launch a new background FFmpeg process.

        If *ts_offset* is given (in seconds), FFmpeg's ``-output_ts_offset``
        is set so timestamps start at that value instead of 0.  The fixed
        encoder priming delay is subtracted automatically.
        Any previously running process is closed first.
        """

        self.close()
        cmd = self._build_command(ts_offset)
        log.debug("Starting ad-replacement filler: %r", cmd)
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )

    def read_duration(self, duration: float) -> bytes:
        """Read approximately *duration* seconds of MPEG-TS filler.

        The byte count is an estimate based on the target bitrate.  Slight
        over/under is harmless because players sync on PTS timestamps.
        """

        if not self._process or self._process.stdout is None:
            return b""

        nbytes = max(_TS_PACKET_SIZE, math.ceil(duration * _FILLER_BYTES_PER_SECOND))
        # Align to TS packet boundary (188 bytes) to avoid splitting packets
        # at the filler/real-stream transition point
        nbytes = ((nbytes + _TS_PACKET_SIZE - 1) // _TS_PACKET_SIZE) * _TS_PACKET_SIZE
        data = self._process.stdout.read(nbytes)

        if not data:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace").strip() if self._process.stderr else ""
                log.warning("FFmpeg filler process exited with code %d%s", self._process.returncode, f": {stderr}" if stderr else "")
                self._process = None
            else:
                log.warning("FFmpeg filler process returned no data")
            return b""

        return data

    def close(self) -> None:
        """Terminate the FFmpeg process and clean up."""

        proc = self._process
        if proc is None:
            return

        self._process = None
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass

    # -- internals -------------------------------------------------------------

    def _build_command(self, ts_offset: float | None = None) -> list[str]:
        w, h, fps = self._width, self._height, self._framerate

        cmd: list[str] = [self._ffmpeg_path, "-y", "-nostats", "-hide_banner", "-loglevel", "error"]

        if self._image:
            # Loop a still image as video source, scale/pad to exact target size
            cmd += ["-loop", "1", "-i", self._image]
        else:
            # Solid black via lavfi
            cmd += ["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps}"]

        # Silent audio source — 48 kHz to match Twitch's AAC configuration
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]

        # Map audio before video so MPEG-TS PID assignment matches Twitch:
        #   PID 0x100 = Audio (AAC), PID 0x101 = Video (H.264)
        # Input 0 is always video (color/image), input 1 is always audio (anullsrc).
        cmd += ["-map", "1:a", "-map", "0:v"]

        # Video filter must come after all inputs (output option, not input option)
        if self._image:
            vfilter = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                f"fps={fps},"
                "format=yuv420p"
            )
            cmd += ["-vf", vfilter]

        # Pixel format as output option (must come after all inputs)
        if not self._image:
            cmd += ["-pix_fmt", "yuv420p"]

        # Encoding — ultrafast for minimal CPU, stillimage tune for static frames
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage"]
        cmd += ["-c:a", "aac", "-b:a", "128k"]

        # Shift output timestamps to match the real stream's position.
        # Subtract the fixed encoder priming delay so the first output
        # frame lands at approximately the requested PTS.
        if ts_offset is not None:
            adjusted = max(0.0, ts_offset - _ENCODER_STARTUP_DELAY)
            cmd += ["-output_ts_offset", f"{adjusted:.6f}"]

        # Output as MPEG-TS to stdout
        cmd += ["-f", "mpegts", "pipe:1"]

        return cmd
