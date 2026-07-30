"""Generate fixtures for the MCP Roots live acceptance tests.

PIL-only — no reportlab dependency. PIL is already a transitive dep of docling.

Produces (under --workdir, default /tmp/docling_roots_live):

    <workdir>/allowed/test_doc.pdf       — 1-page raster PDF, OCR + table-structure exercise
    <workdir>/allowed/test_table.png     — same image as a standalone PNG
    <workdir>/forbidden/test_doc.pdf     — same PDF in a non-allowed dir (rejection target)

Usage:
    python scripts/live_tests/make_fixtures.py
    python scripts/live_tests/make_fixtures.py --workdir /some/other/path
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    """Best-effort load a TrueType font for the labels; fall back to PIL default."""
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_table_image() -> Image.Image:
    """Render one page-sized image with a heading and a 3x3 table."""
    width, height = 1240, 900  # ~ letter @ 150 dpi
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_title = _load_font(48)
    font_body = _load_font(28)
    font_head = _load_font(32)

    draw.text(
        (80, 60),
        "Docling MCP Roots - live acceptance fixture",
        fill="black",
        font=font_title,
    )
    draw.text(
        (80, 140),
        "This page exercises the OCR + table-structure pipeline.",
        fill="black",
        font=font_body,
    )

    rows = [
        ("Trial", "Method", "Expected Outcome"),
        ("1", "Static --allowed-directories", "pass for path inside root"),
        ("2", "Static --allowed-directories", "reject for path outside"),
        ("3", "Client Roots (Desktop / Code)", "refresh + pass after roots msg"),
    ]
    cols_x = [80, 240, 760, 1160]
    rows_y = [260, 360, 460, 560, 660]
    for x in cols_x:
        draw.line([(x, rows_y[0]), (x, rows_y[-1])], fill="black", width=4)
    for y in rows_y:
        draw.line([(cols_x[0], y), (cols_x[-1], y)], fill="black", width=4)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            draw.text(
                (cols_x[c] + 16, rows_y[r] + 22),
                cell,
                fill="black",
                font=font_head if r == 0 else font_body,
            )
    return img


def main(workdir: Path) -> None:
    """Generate the allowed/ and forbidden/ fixture trees under ``workdir``."""
    allowed = workdir / "allowed"
    forbidden = workdir / "forbidden"
    allowed.mkdir(parents=True, exist_ok=True)
    forbidden.mkdir(parents=True, exist_ok=True)

    pdf_inside = allowed / "test_doc.pdf"
    png_inside = allowed / "test_table.png"
    pdf_outside = forbidden / "test_doc.pdf"

    img = make_table_image()
    img.save(png_inside, "PNG")
    img.convert("RGB").save(pdf_inside, "PDF", resolution=150.0)
    shutil.copy(pdf_inside, pdf_outside)

    for p in (pdf_inside, png_inside, pdf_outside):
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/docling_roots_live"),
        help="Directory to write fixtures into (default: /tmp/docling_roots_live)",
    )
    args = parser.parse_args()
    main(args.workdir)
