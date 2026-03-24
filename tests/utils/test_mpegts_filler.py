from __future__ import annotations

import math
import os
import subprocess
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from streamlink.utils.mpegts_filler import (
    FFmpegFiller,
    _ENCODER_STARTUP_DELAY,
    _FILLER_BYTES_PER_SECOND,
    _TS_CLOCK_RATE,
    extract_last_pts,
)


class TestFFmpegFillerCommand:
    """Test that FFmpegFiller builds the correct ffmpeg command."""

    def test_black_frame_command(self):
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        cmd = filler._build_command()

        assert cmd[0] == "/usr/bin/ffmpeg"
        assert "-f" in cmd
        assert "lavfi" in cmd
        assert "color=c=black:s=1920x1080:r=30.0" in cmd
        assert "anullsrc=r=48000:cl=stereo" in cmd
        assert "libx264" in cmd
        assert "aac" in cmd
        assert cmd[-3:-1] == ["-f", "mpegts"]
        assert cmd[-1] == "pipe:1"
        # Audio mapped before video for PID compatibility with Twitch MPEG-TS
        map_indices = [i for i, v in enumerate(cmd) if v == "-map"]
        assert len(map_indices) == 2
        assert cmd[map_indices[0] + 1] == "1:a"
        assert cmd[map_indices[1] + 1] == "0:v"
        # No ts_offset by default
        assert "-output_ts_offset" not in cmd

    def test_black_frame_custom_resolution(self):
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1280, 720, 60.0)
        cmd = filler._build_command()

        assert "color=c=black:s=1280x720:r=60.0" in cmd

    def test_ts_offset(self):
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        cmd = filler._build_command(ts_offset=34298.697)

        assert "-output_ts_offset" in cmd
        offset_idx = cmd.index("-output_ts_offset")
        # Should subtract the encoder startup delay
        expected = f"{34298.697 - _ENCODER_STARTUP_DELAY:.6f}"
        assert cmd[offset_idx + 1] == expected

    def test_ts_offset_small_value(self):
        """When ts_offset is smaller than startup delay, clamp to 0."""
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        cmd = filler._build_command(ts_offset=0.5)

        offset_idx = cmd.index("-output_ts_offset")
        assert cmd[offset_idx + 1] == "0.000000"

    def test_image_command(self):
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0, image="/tmp/logo.png")
        cmd = filler._build_command()

        assert "-loop" in cmd
        assert "/tmp/logo.png" in cmd
        # Should have a scale/pad video filter after all inputs
        vf_idx = cmd.index("-vf")
        vfilter = cmd[vf_idx + 1]
        assert "scale=1920:1080" in vfilter
        assert "pad=1920:1080" in vfilter
        assert "fps=30.0" in vfilter
        # -vf must come after -map (i.e. after all inputs)
        map_indices = [i for i, v in enumerate(cmd) if v == "-map"]
        assert vf_idx > map_indices[-1]
        # No lavfi color source for video
        # (lavfi still used for audio)
        color_inputs = [i for i, v in enumerate(cmd) if "color=c=black" in str(v)]
        assert len(color_inputs) == 0
        # Image command: audio mapped before video for PID compatibility
        map_indices = [i for i, v in enumerate(cmd) if v == "-map"]
        assert len(map_indices) == 2
        assert cmd[map_indices[0] + 1] == "1:a"
        assert cmd[map_indices[1] + 1] == "0:v"

    def test_image_from_pathlib(self, tmp_path):
        img = tmp_path / "overlay.png"
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0, image=img)
        cmd = filler._build_command()
        assert str(img) in cmd


class TestFFmpegFillerLifecycle:
    """Test start/read/close with mocked subprocess."""

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_start(self, mock_popen):
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.start()

        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert cmd[0] == "/usr/bin/ffmpeg"
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        assert kwargs["stdin"] == subprocess.DEVNULL

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_start_with_offset(self, mock_popen):
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.start(ts_offset=12345.678)

        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert "-output_ts_offset" in cmd
        idx = cmd.index("-output_ts_offset")
        expected = f"{12345.678 - _ENCODER_STARTUP_DELAY:.6f}"
        assert cmd[idx + 1] == expected

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_start_closes_previous(self, mock_popen):
        mock_proc1 = MagicMock()
        mock_proc2 = MagicMock()
        mock_popen.side_effect = [mock_proc1, mock_proc2]

        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.start()
        filler.start(ts_offset=100.0)

        # First process should be killed
        mock_proc1.kill.assert_called_once()
        assert mock_popen.call_count == 2

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_read_duration(self, mock_popen):
        mock_proc = MagicMock()
        fake_data = b"\x47" * 18800
        mock_proc.stdout.read.return_value = fake_data
        mock_popen.return_value = mock_proc

        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.start()

        data = filler.read_duration(2.0)

        raw = 2.0 * _FILLER_BYTES_PER_SECOND
        expected_bytes = ((math.ceil(raw) + 187) // 188) * 188
        mock_proc.stdout.read.assert_called_once()
        read_size = mock_proc.stdout.read.call_args[0][0]
        assert read_size == expected_bytes
        assert read_size % 188 == 0
        assert data == fake_data

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_read_duration_minimum_packet(self, mock_popen):
        """Even for very short durations, we read at least 188 bytes (one TS packet)."""
        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b"\x47" * 188
        mock_popen.return_value = mock_proc

        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.start()

        filler.read_duration(0.0001)
        read_size = mock_proc.stdout.read.call_args[0][0]
        assert read_size == 188

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_read_duration_no_process(self, mock_popen):
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        assert filler.read_duration(1.0) == b""

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_read_duration_empty_data_process_exited(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b""
        mock_proc.poll.return_value = 1
        mock_proc.stderr.read.return_value = b"some error"
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.start()

        data = filler.read_duration(1.0)
        assert data == b""
        assert filler._process is None

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_read_duration_empty_data_process_alive(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout.read.return_value = b""
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.start()

        data = filler.read_duration(1.0)
        assert data == b""
        assert filler._process is not None

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_close(self, mock_popen):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.start()
        filler.close()

        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert filler._process is None

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_close_idempotent(self, mock_popen):
        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.close()
        filler.close()

    @patch("streamlink.utils.mpegts_filler.subprocess.Popen")
    def test_close_handles_errors(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.kill.side_effect = OSError("already dead")
        mock_popen.return_value = mock_proc

        filler = FFmpegFiller("/usr/bin/ffmpeg", 1920, 1080, 30.0)
        filler.start()
        filler.close()


class TestExtractLastPts:
    """Test PTS extraction from MPEG-TS data."""

    def test_no_data(self):
        assert extract_last_pts(b"") is None

    def test_no_pts_in_data(self):
        data = b"\x47" + b"\x00" * 187
        assert extract_last_pts(data) is None

    def test_extract_from_real_ts(self):
        """Extract PTS from a real MPEG-TS file if available."""

        ts_path = os.path.join(os.path.dirname(__file__), "..", "..", "fps_shaka_stream.ts")
        if not os.path.exists(ts_path):
            pytest.skip("Test TS file not available")

        with open(ts_path, "rb") as f:
            data = f.read(188 * 100)  # Read first 100 packets

        pts = extract_last_pts(data)
        assert pts is not None
        assert pts > 0
        # Convert to seconds — should be in a reasonable range
        assert pts / _TS_CLOCK_RATE > 0
