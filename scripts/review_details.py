#!/usr/bin/env python3
"""Review sheets: original PNG vs 19-color study vs numbered black/white SVG."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

def font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()

def main(source, output):
    source = Image.open(source).convert("RGB")
    color = Image.open(output / "color_segmentation_review.png").convert("RGB")
    outline = Image.open(output / "lineart_preview.png").convert("RGB")
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    w, h = source.size
    ow, oh = outline.size
    sheet_mm_w, sheet_mm_h = audit["rendered_page_mm"]
    art_mm_w, art_mm_h = audit["artwork_size_mm"]
    xoff_mm = (sheet_mm_w - art_mm_w) / 2
    yoff_mm = (sheet_mm_h - art_mm_h) / 2
    # The original artwork is scaled uniformly and centered on the print page.
    scale = ow * art_mm_w / sheet_mm_w / w
    xoff = xoff_mm * ow / sheet_mm_w
    yoff = yoff_mm * oh / sheet_mm_h

    regions = [
        ("horizon", (0, 0, 1122, 335)),
        ("face_hairpin", (265, 185, 710, 570)),
        ("kimono_sakura", (175, 520, 755, 1035)),
        ("foreground_lantern", (655, 890, 1110, 1390))
    ]
    panel_w = 1000
    heading_h = 63
    margin = 20
    for name, coords in regions:
        x0,y0,x1,y1 = coords
        coords = (max(0,x0),max(0,y0),min(w,x1),min(h,y1))
        box_float = (
            xoff + coords[0]*scale,
            yoff + coords[1]*scale,
            xoff + coords[2]*scale,
            yoff + coords[3]*scale
        )
        outline_box = tuple(round(v) for v in box_float)
        sources = [
            source.crop(coords),
            color.crop(coords),
            outline.crop(outline_box)
        ]
        target_height = round(panel_w * sources[0].height / sources[0].width)
        canvas = Image.new(
            "RGB", (panel_w+2*margin,3*(target_height+heading_h)+margin),
            (249,247,242))
        pen = ImageDraw.Draw(canvas)
        sections = [
            "ИСХОДНИК — БЕЗ ИЗМЕНЕНИЙ",
            f"ЦВЕТНАЯ СЕГМЕНТАЦИЯ — 19 ЦВЕТОВ / {audit['regions_total']} ОБЛАСТЕЙ",
            "ЧЁРНО-БЕЛЫЕ КОНТУРЫ — 0,2 ММ / НОМЕРА 5,3 PT"
        ]
        for i,(title,picture) in enumerate(zip(sections,sources)):
            top=i*(target_height+heading_h)
            pen.text((margin+4,top+13),title,font=font(27),fill=(32,32,37))
            scaled=picture.resize((panel_w,target_height),Image.Resampling.LANCZOS)
            canvas.paste(scaled,(margin,top+heading_h))
        canvas.save(output / f"comparison_{name}.png",optimize=True)
        print("Created",name)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    main(a.input,a.out)
