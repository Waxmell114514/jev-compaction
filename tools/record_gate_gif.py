"""Re-record docs/gate.gif from docs/index.html.

The GIF is a binary copy of the animation, so it goes stale the moment the
animation changes. This is the recipe that made it.

Not part of [dev] -- it needs a browser and an image library the rest of the
repo is deliberately free of:

    uv pip install --python .venv/bin/python playwright pillow
    .venv/bin/playwright install chromium
    .venv/bin/python tools/record_gate_gif.py

ffmpeg comes from Playwright's own browser bundle, so there is nothing else
to install; it is a cut-down build with no gif muxer, which is why the frames
go through Pillow rather than straight out of ffmpeg.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
import time

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "index.html"
OUT = ROOT / "docs" / "gate.gif"

WIDTH = 880      # rendered width of the gif
FPS = 10
COLORS = 96


def _ffmpeg() -> str:
    hits = sorted(pathlib.Path("/opt/pw-browsers").glob("ffmpeg-*/ffmpeg-linux"))
    if not hits:
        import playwright  # noqa: PLC0415 -- only needed on the fallback path

        base = pathlib.Path(playwright.__file__).parent / "driver" / "package" / ".local-browsers"
        hits = sorted(base.glob("ffmpeg-*/ffmpeg-linux"))
    if not hits:
        raise SystemExit("no ffmpeg in the Playwright bundle")
    return str(hits[-1])


def record(dest: pathlib.Path) -> tuple[pathlib.Path, float]:
    """Play one full loop and return the webm plus how long the loop took."""
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--force-color-profile=srgb"])
        ctx = browser.new_context(
            viewport={"width": 1200, "height": 820},
            record_video_dir=str(dest),
            record_video_size={"width": 1200, "height": 820},
        )
        page = ctx.new_page()
        page.goto(PAGE.as_uri())
        page.wait_for_timeout(600)

        # The page loops forever, so stop when the turn counter comes back
        # round to 1 having been all the way to 4.
        seen: list[str] = []
        last, start = None, time.time()
        while time.time() - start < 120:
            now = page.evaluate("document.getElementById('in-meta')?.textContent || ''")
            if now != last:
                seen.append(now.strip())
                last = now
                if len(seen) > 1 and seen[-1] == "turn 1" and "turn 4" in seen:
                    break
            page.wait_for_timeout(100)
        loop = time.time() - start
        ctx.close()
        browser.close()
    return next(dest.glob("*.webm")), loop


def to_gif(webm: pathlib.Path, loop: float, frames_dir: pathlib.Path) -> None:
    subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-ss", "0.35", "-t", f"{loop - 1.2:.2f}", "-i", str(webm),
         "-vf", f"scale={WIDTH}:-2", "-r", str(FPS), "-c:v", "png",
         str(frames_dir / "f%04d.png")],
        check=True,
    )
    files = sorted(frames_dir.glob("f*.png"))

    # One palette for the whole clip: a GIF only stays small if consecutive
    # frames index into the same table, and this page barely changes between
    # frames. Dithering would destroy that, so it stays off.
    sample = [Image.open(f).convert("RGB") for f in files[:: max(1, len(files) // 24)]]
    w, h = sample[0].size
    strip = Image.new("RGB", (w, h * len(sample)))
    for i, im in enumerate(sample):
        strip.paste(im, (0, i * h))
    palette = strip.quantize(colors=COLORS, method=Image.Quantize.MEDIANCUT)

    shots = [
        Image.open(f).convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
        for f in files
    ]
    shots[0].save(OUT, save_all=True, append_images=shots[1:], duration=1000 // FPS,
                  loop=0, optimize=True, disposal=1)
    print(f"{OUT.relative_to(ROOT)}: {len(files)} frames, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        video, seconds = record(tmp_path)
        frames = tmp_path / "frames"
        frames.mkdir()
        to_gif(video, seconds, frames)
