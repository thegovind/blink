"""Convert Playwright's real-time recording without altering its frame timestamps."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run(*args: str) -> None:
    from imageio_ffmpeg import get_ffmpeg_exe

    command = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {result.stderr[-1500:]}")


def normalize_capture(capture: Path, output: Path, *, viewport: dict[str, int],
                      scale: float) -> None:
    """Remove Playwright's empty padding around a high-DPI CSS viewport, retaining frame PTS."""
    width, height = viewport["width"], viewport["height"]
    size = (round(width * scale), round(height * scale))
    _run("-i", str(capture), "-vf",
         f"crop={width}:{height}:0:0,scale={size[0]}:{size[1]}:flags=lanczos,setsar=1",
         "-fps_mode", "passthrough", "-c:v", "libvpx", "-deadline", "realtime",
         "-cpu-used", "5", "-b:v", "1800k", "-pix_fmt", "yuv420p", "-an",
         "-threads", "2", str(output))
    _require_output(output)


def _require_output(path: Path) -> None:
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"ffmpeg did not produce {path}")


def poster_frame(webm: Path, poster: Path) -> None:
    """Take a decision frame if present; very short episodes fall back to the first frame."""
    for second in (2, 1, 0):
        poster.unlink(missing_ok=True)
        try:
            _run("-ss", str(second), "-i", str(webm), "-frames:v", "1",
                 "-threads", "2", str(poster))
        except RuntimeError:
            if second == 0:
                raise
            continue
        if poster.is_file() and poster.stat().st_size:
            return
    raise RuntimeError(f"ffmpeg could not extract any frame from {webm}")


def convert(webm: Path) -> dict[str, str]:
    """Write an H.264 MP4, a short small GIF, and a poster beside video.webm."""
    webm = Path(webm)
    if not webm.is_file():
        raise FileNotFoundError(webm)
    mp4 = webm.with_suffix(".mp4")
    gif = webm.with_name("preview.gif")
    poster = webm.with_name("poster.png")
    _run("-i", str(webm), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
         "-threads", "2", str(mp4))
    _require_output(mp4)
    _run("-t", "8", "-i", str(webm), "-filter_complex",
         "fps=6,scale=480:-2:flags=lanczos,split[a][b];"
         "[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer",
         "-loop", "0", "-threads", "2", str(gif))
    _require_output(gif)
    poster_frame(webm, poster)
    return {"webm": webm.name, "mp4": mp4.name, "gif": gif.name, "poster": poster.name}
