#!/usr/bin/env python3
"""Render dark almond pupils as preprinted line art, not miniature paint cells."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
import cv2
from PIL import Image
import cairosvg

HERE=Path("face_v7")
HERE.mkdir(exist_ok=True)
NS="{http://www.w3.org/2000/svg}"
SCALE=400/1122
YOFF=(600-1402*SCALE)/2
XOFF=0
PUPILS=[
  ("pupil_near",[(582,391),(587,389),(590,394),(587,400),(583,401),(581,397)]),
  ("pupil_far",[(654,386),(659,385),(660,390),(658,397),(654,396),(652,391)])
]
def svg_pupil(p):
    def point(v):return f"{XOFF+(v[0]+.5)*SCALE:.3f},{YOFF+(v[1]+.5)*SCALE:.3f}"
    return 'M'+point(p[0])+''.join('L'+point(q) for q in p[1:])+'Z'
def main():
    image=np.asarray(Image.open(HERE/"color_artist_preview.png").convert("RGB")).copy()
    for name,p in PUPILS:
        cv2.fillPoly(image,[np.asarray(p,np.int32)],(24,16,27),cv2.LINE_8)
    Image.fromarray(image).save(HERE/"color_artist_with_ink.png")
    tree=ET.parse(HERE/"lineart_review.svg")
    root=tree.getroot()
    groups=list(root.iter(NS+"g"))
    ink=next((g for g in groups if g.attrib.get("id")=="artist_face_ink"),None)
    if ink is None:raise RuntimeError("Missing existing artistic facial ink group")
    for name,p in PUPILS:
        ET.SubElement(ink,NS+"path",attrib={"id":name,
          "d":svg_pupil(p),"fill":"#18101B","stroke":"none"})
    ET.register_namespace("","http://www.w3.org/2000/svg")
    out=HERE/"lineart_with_pupils.svg"
    tree.write(out,encoding="utf-8",xml_declaration=True)
    cairosvg.svg2png(url=str(out),write_to=str(HERE/"lineart_with_pupils.png"),output_width=2400,output_height=3600)
    cairosvg.svg2png(url=str(out),write_to=str(HERE/"lineart_with_pupils_screen.png"),output_width=1200,output_height=1800)
    print(json.dumps({"pupils":"two source-positioned filled dark slit details",
                      "paint_cell_colors_unchanged":True,
                      "printed_black_pupils":"not additional paintable regions"}))
if __name__=="__main__":main()
